# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of AlignmentStreamAnalyzer for T3 TTS model.
Ported from PyTorch version in alignment_stream_analyzer.py

This module analyzes attention patterns during generation to detect:
- False starts (hallucinations at beginning)
- Long tails (generation continuing after text completion)
- Alignment repetition (text being repeated)
- Token repetition (same token generated multiple times)

When issues are detected, it modifies logits to force EOS token generation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# Attention heads that show good text-speech alignment
# Format: (layer_idx, head_idx)
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResultMLX:
    """Result from alignment analysis at each step."""
    # Was this frame detected as being part of a noisy beginning with potential hallucinations?
    false_start: bool
    # Was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # Was this frame detected as repeating existing text content?
    repetition: bool
    # Was the alignment position too far from the previous frame?
    discontinuity: bool
    # Has inference reached the end of the text tokens?
    complete: bool
    # Approximate position in the text token sequence
    position: int


class AlignmentStreamAnalyzerMLX:
    """
    MLX implementation of alignment stream analyzer for T3 TTS.
    
    Analyzes attention patterns to detect generation issues and force EOS when needed.
    Unlike PyTorch version, this doesn't use hooks - attention is passed explicitly.
    """
    
    def __init__(
        self,
        text_tokens_slice: Tuple[int, int],
        eos_idx: int = 6562,
    ):
        """
        Initialize the alignment stream analyzer.
        
        Args:
            text_tokens_slice: (start, end) indices of text tokens in the sequence
            eos_idx: Token ID for end-of-speech (default: 6562)
        """
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        
        # Alignment matrix: (num_speech_frames, num_text_tokens)
        self.alignment = mx.zeros((0, j - i))
        
        # State tracking
        self.curr_frame_pos = 0
        self.text_position = 0
        
        self.started = False
        self.started_at: Optional[int] = None
        
        self.complete = False
        self.completed_at: Optional[int] = None
        
        # Token tracking for repetition detection
        self.generated_tokens: List[int] = []
        
        # Attention buffers for each aligned head
        self.last_aligned_attns: List[Optional[mx.array]] = [None] * len(LLAMA_ALIGNED_HEADS)
    
    def update_attention(self, attn_weights: mx.array, buffer_idx: int):
        """
        Update attention weights for a specific aligned head.
        
        Called after each forward pass with attention from the aligned heads.
        
        Args:
            attn_weights: Attention weights from one head, shape depends on step
            buffer_idx: Index into LLAMA_ALIGNED_HEADS (0, 1, or 2)
        """
        self.last_aligned_attns[buffer_idx] = attn_weights
    
    def step(self, logits: mx.array, next_token: Optional[int] = None) -> mx.array:
        """
        Analyze alignment and potentially force EOS by modifying logits.
        
        Args:
            logits: Current logits from model, shape (1, vocab_size)
            next_token: The last generated token (for repetition tracking)
            
        Returns:
            Modified logits (may force EOS if issues detected)
        """
        i, j = self.text_tokens_slice
        S = j - i  # Number of text tokens
        
        # Track generated tokens for repetition detection
        if next_token is not None:
            self.generated_tokens.append(next_token)
            # Keep only last 8 tokens to prevent memory issues
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]
        
        # Check for token repetition (2+ same tokens in a row)
        # Only trigger after we've generated at least 20 tokens (avoid false positives at start)
        # and require at least 3 tokens in history to check
        min_tokens_before_repetition_check = 20
        token_repetition = (
            len(self.generated_tokens) >= 3 and
            self.curr_frame_pos >= min_tokens_before_repetition_check and
            len(set(self.generated_tokens[-2:])) == 1
        )
        
        if token_repetition:
            repeated_token = self.generated_tokens[-1]
            logger.warning(f"🚨 Detected 2x repetition of token {repeated_token}")
        
        # Process attention if available
        long_tail = False
        alignment_repetition = False
        
        if all(a is not None for a in self.last_aligned_attns):
            # Stack and average attention from aligned heads
            aligned_attn = mx.stack(self.last_aligned_attns, axis=0)
            aligned_attn = mx.mean(aligned_attn, axis=0)  # (N, N) or (1, N)
            
            # Extract alignment chunk for text tokens
            if self.curr_frame_pos == 0:
                # First chunk has conditioning info, text tokens, and BOS token
                # Shape: (T, S) where T = speech frames in first chunk
                if aligned_attn.ndim >= 2 and aligned_attn.shape[0] > j:
                    A_chunk = aligned_attn[j:, i:j]
                else:
                    A_chunk = mx.zeros((1, S))
            else:
                # Subsequent chunks have 1 frame due to KV-caching
                # Shape: (1, S)
                if aligned_attn.ndim >= 2:
                    A_chunk = aligned_attn[:, i:j]
                else:
                    A_chunk = mx.zeros((1, S))
            
            # Ensure A_chunk is 2D
            if A_chunk.ndim == 1:
                A_chunk = mx.expand_dims(A_chunk, axis=0)
            
            # Apply monotonic masking - zero out future positions
            if A_chunk.shape[1] > self.curr_frame_pos + 1:
                mask = mx.concatenate([
                    mx.ones((A_chunk.shape[0], self.curr_frame_pos + 1)),
                    mx.zeros((A_chunk.shape[0], A_chunk.shape[1] - self.curr_frame_pos - 1))
                ], axis=1)
                A_chunk = A_chunk * mask
            
            # Concatenate to alignment history
            self.alignment = mx.concatenate([self.alignment, A_chunk], axis=0)
            
            A = self.alignment
            T = A.shape[0]  # Total speech frames so far
            
            # Update text position from alignment
            if A_chunk.shape[0] > 0:
                cur_text_posn = int(mx.argmax(A_chunk[-1]))
                discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
                if not discontinuity:
                    self.text_position = cur_text_posn
            
            # Detect false starts (hallucinations at beginning)
            if T >= 2 and not self.started:
                last_2_corner = float(mx.max(A[-2:, -2:]))
                first_4_max = float(mx.max(A[:, :4])) if A.shape[1] >= 4 else 0.5
                false_start = last_2_corner > 0.1 or first_4_max < 0.5
                self.started = not false_start
                if self.started and self.started_at is None:
                    self.started_at = T
            
            # Is generation likely complete?
            self.complete = self.complete or self.text_position >= S - 3
            if self.complete and self.completed_at is None:
                self.completed_at = T
            
            # Detect long tail (activations for final tokens lasting too long)
            if self.complete and self.completed_at is not None:
                tail_activation = A[self.completed_at:, -3:]
                if tail_activation.size > 0:
                    tail_sum = float(mx.sum(tail_activation, axis=0).max())
                    long_tail = tail_sum >= 5  # ~200ms worth
            
            # Detect alignment repetition
            if self.complete and self.completed_at is not None:
                earlier_activation = A[self.completed_at:, :-5]
                if earlier_activation.size > 0:
                    max_per_row = mx.max(earlier_activation, axis=1)
                    alignment_repetition = float(mx.sum(max_per_row)) > 5
        
        # Suppress EOS if text generation isn't complete yet
        if self.text_position < S - 3 and S > 5:
            # Create new logits with EOS suppressed
            logits_list = list(logits.flatten())
            logits_list[self.eos_idx] = -32768.0  # -2^15
            logits = mx.array(logits_list).reshape(logits.shape)
        
        # Force EOS if bad ending detected
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            # Set all logits to very negative, except EOS
            forced_logits = mx.full(logits.shape, -32768.0)
            # Use index update for EOS position
            forced_logits_flat = list(forced_logits.flatten())
            forced_logits_flat[self.eos_idx] = 32768.0  # 2^15
            logits = mx.array(forced_logits_flat).reshape(logits.shape)
        
        self.curr_frame_pos += 1
        return logits
    
    def reset(self):
        """Reset the analyzer state between inference calls."""
        i, j = self.text_tokens_slice
        self.alignment = mx.zeros((0, j - i))
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        self.generated_tokens = []
        self.last_aligned_attns = [None] * len(LLAMA_ALIGNED_HEADS)


class SimpleTokenRepetitionDetector:
    """
    Simplified detector that only checks for token repetition.
    
    Use this when full attention-based analysis isn't available.
    This is a lightweight alternative that catches the most common issue.
    """
    
    def __init__(self, eos_idx: int = 6562, repetition_threshold: int = 2):
        """
        Initialize the simple detector.
        
        Args:
            eos_idx: Token ID for end-of-speech
            repetition_threshold: Number of repeated tokens to trigger EOS
        """
        self.eos_idx = eos_idx
        self.repetition_threshold = repetition_threshold
        self.generated_tokens: List[int] = []
    
    def step(self, logits: mx.array, next_token: int) -> Tuple[mx.array, bool]:
        """
        Check for token repetition and potentially force EOS.
        
        Args:
            logits: Current logits from model
            next_token: The token that was just generated
            
        Returns:
            Tuple of (potentially modified logits, should_stop)
        """
        self.generated_tokens.append(next_token)
        
        # Keep only recent tokens
        if len(self.generated_tokens) > 10:
            self.generated_tokens = self.generated_tokens[-10:]
        
        # Check for repetition
        if len(self.generated_tokens) >= self.repetition_threshold:
            recent = self.generated_tokens[-self.repetition_threshold:]
            if len(set(recent)) == 1:
                logger.warning(f"🚨 Token {recent[0]} repeated {self.repetition_threshold}x, forcing EOS")
                # Force EOS
                forced_logits = mx.full(logits.shape, -32768.0)
                forced_logits_flat = list(forced_logits.flatten())
                forced_logits_flat[self.eos_idx] = 32768.0
                return mx.array(forced_logits_flat).reshape(logits.shape), True
        
        return logits, False
    
    def reset(self):
        """Reset state between generations."""
        self.generated_tokens = []
