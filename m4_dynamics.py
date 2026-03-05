import numpy as np
import hashlib
import time
import os

class M4Dynamics:
    """
    M4Dynamics: The dynamic cryptographic protection and chaos management module.
    Handles the spatial dislocation of the Operator and Residual matrices, utilizing 
    SHA-256 and unique IVs (Nonces) to prevent Known-Plaintext Attacks (KPA).
    """

    def __init__(self, master_seed="MND_TOP_SECRET"):
        """
        Initializes the dynamics module with a master cryptographic key.
        """
        self.master_seed = master_seed

    def _generate_chaos_sequence(self, seed, length):
        """
        Generates a deterministic chaotic sequence based on cascaded SHA-256 hashes.
        This ensures that even a 1-bit change in the seed triggers a complete avalanche effect.
        
        :param seed: The composite string (Master/Guest + Nonce).
        :param length: Required length of the sequence.
        :return: A deterministic array of pseudo-random integers.
        """
        sequence = []
        salt = "chaos_salt_v1"
        current_hash = hashlib.sha256(f"{seed}{salt}".encode()).digest()
        
        while len(sequence) < length:
            for byte in current_hash:
                sequence.append(byte)
                if len(sequence) >= length:
                    break
            # Cascade the hash for infinite deterministic entropy
            current_hash = hashlib.sha256(current_hash).digest()
            
        return np.array(sequence, dtype=np.uint32)

    def generate_guest_seed(self, valid_duration_minutes=60):
        """
        Generates a time-based temporary Guest Seed (similar to TOTP).
        Ensures temporal access control without sharing the Master Seed.
        """
        timestamp = int(time.time() // (valid_duration_minutes * 60))
        raw_string = f"{self.master_seed}_{timestamp}"
        guest_seed = hashlib.sha256(raw_string.encode()).hexdigest()
        return guest_seed

    def generate_file_nonce(self):
        """
        Generates a 16-byte cryptographically secure random Nonce (Initialization Vector).
        Must be generated uniquely for every single file/session to prevent pattern recognition.
        """
        return os.urandom(16).hex()

    # --- RESIDUAL (THE ASH) FUNCTIONS ---

    def scramble_residual(self, residual_matrix, seed, nonce=""):
        """
        Spatially dislocates (scrambles) the Residual matrix using a deterministic permutation.
        The permutation key is a composite of the Seed and the unique file Nonce.
        """
        original_shape = residual_matrix.shape
        flat = residual_matrix.flatten()
        n = len(flat)

        # Composite seed prevents identical pattern generation for identical files
        res_seed = f"{seed}_residual_{nonce}" 
        chaos_seq = self._generate_chaos_sequence(res_seed, n)
        
        # Argsort creates a perfect 1:1 reversible permutation mapping
        perm_indices = np.argsort(chaos_seq)
        shuffled_flat = flat[perm_indices]

        return shuffled_flat.reshape(original_shape)

    def descramble_residual(self, scrambled_residual, seed, nonce=""):
        """
        Restores the exact original coordinates of the Residual matrix using inverse permutation.
        """
        original_shape = scrambled_residual.shape
        flat_scrambled = scrambled_residual.flatten()
        n = len(flat_scrambled)

        # Must use the exact same composite seed for identical chaos regeneration
        res_seed = f"{seed}_residual_{nonce}"
        chaos_seq = self._generate_chaos_sequence(res_seed, n)
        perm_indices = np.argsort(chaos_seq)

        # Reconstruct the original layout via coordinate inversion
        restored_flat = np.zeros_like(flat_scrambled)
        restored_flat[perm_indices] = flat_scrambled

        return restored_flat.reshape(original_shape)

    # --- SPECTRAL COMPONENT (THE OPERATOR) FUNCTIONS ---

    def scramble_operator(self, spectral_component, seed, nonce=""):
        """
        Scrambles the Spectral Operator (U, Sigma, V^T) matrices.
        Permutes the spatial coordinates of U and V^T, and applies a deterministic 
        micro-mask to the Singular Values (Sigma) to destroy structural readability.
        """
        Uk, S, Vtk = spectral_component
        
        op_seed = f"{seed}_operator_{nonce}"
        total_len = Uk.shape[0] + Vtk.shape[1] + len(S)
        chaos_seq = self._generate_chaos_sequence(op_seed, total_len * 2)
        
        # Permute the U matrix rows
        perm_u = np.argsort(chaos_seq[:Uk.shape[0]])
        Uk_scrambled = Uk[perm_u, :]
        
        # Permute the V^T matrix columns
        perm_vt = np.argsort(chaos_seq[Uk.shape[0]:Uk.shape[0]+Vtk.shape[1]])
        Vtk_scrambled = Vtk[:, perm_vt]
        
        # Apply a reversible floating-point mask to the Singular Values
        mask_s = chaos_seq[-len(S):] % 1000
        S_scrambled = S + (mask_s * 0.001)
        
        return (Uk_scrambled, S_scrambled, Vtk_scrambled)

    def descramble_operator(self, scrambled_component, seed, nonce=""):
        """
        Reconstructs the Spectral Operator by reversing the spatial dislocation 
        and safely removing the floating-point mask from the Singular Values.
        """
        Uk_scr, S_scr, Vtk_scr = scrambled_component
        
        op_seed = f"{seed}_operator_{nonce}"
        
        total_len = Uk_scr.shape[0] + Vtk_scr.shape[1] + len(S_scr)
        chaos_seq = self._generate_chaos_sequence(op_seed, total_len * 2)
        
        perm_u = np.argsort(chaos_seq[:Uk_scr.shape[0]])
        perm_vt = np.argsort(chaos_seq[Uk_scr.shape[0]:Uk_scr.shape[0]+Vtk_scr.shape[1]])
        mask_s = chaos_seq[-len(S_scr):] % 1000
        
        # Restore U matrix
        Uk_restored = np.zeros_like(Uk_scr)
        Uk_restored[perm_u, :] = Uk_scr
        
        # Restore V^T matrix
        Vtk_restored = np.zeros_like(Vtk_scr)
        Vtk_restored[:, perm_vt] = Vtk_scr
        
        # Remove the mask to restore precise Singular Values
        S_restored = S_scr - (mask_s * 0.001)
        
        return (Uk_restored, S_restored, Vtk_restored)