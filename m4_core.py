import numpy as np
from PIL import Image

class M4Core:
    """
    M4Core: The mathematical engine of the M4 Architecture.
    Handles the SVD bifurcation process, data block preparation, 
    and ensures 100% lossless data integrity via modulo arithmetic.
    """

    def __init__(self, rank_k=50, matrix_size=(256, 256)):
        """
        Initializes the M4 Core processor.
        
        :param rank_k: The number of singular values retained (defines the detail level of the Operator skeleton).
        :param matrix_size: The fixed block size for data processing (default is 256x256).
        """
        self.rank_k = rank_k
        self.matrix_size = matrix_size

    def prepare_image(self, image_path):
        """
        Loads an image, converts it to grayscale (8-bit), and resizes it 
        to fit the standard matrix block dimension.
        """
        try:
            img = Image.open(image_path).convert('L')  # L = 8-bit pixels, black and white
            img = img.resize(self.matrix_size)
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"Image loading error: {e}")
            return None

    def prepare_text(self, text):
        """
        Converts a raw text string into a square byte matrix, 
        padding with zeros to fill the fixed block size.
        """
        byte_data = text.encode('utf-8')
        total_pixels = self.matrix_size[0] * self.matrix_size[1]
        
        # Truncate text if it exceeds the block size (in production, data is split into multiple blocks)
        if len(byte_data) > total_pixels:
            byte_data = byte_data[:total_pixels]
            
        # Create an empty matrix and populate it with byte data
        flat_array = np.zeros(total_pixels, dtype=np.uint8)
        flat_array[:len(byte_data)] = np.frombuffer(byte_data, dtype=np.uint8)
        
        return flat_array.reshape(self.matrix_size)

    def bifurcate(self, matrix):
        """
        The primary Bifurcation process (Data Splitting).
        
        :param matrix: The original 2D data matrix (Original).
        :return: A tuple containing the Spectral Component (Operator) and the Residual Matrix (The Ash).
        """
        # 1. SVD Decomposition (Singular Value Decomposition)
        matrix_float = matrix.astype(float)
        U, S, Vt = np.linalg.svd(matrix_float, full_matrices=False)

        # 2. Truncation to Rank K (Forming the Operator / Skeleton)
        Uk = U[:, :self.rank_k]
        Sk = np.diag(S[:self.rank_k])
        Vtk = Vt[:self.rank_k, :]

        # 3. Create the mathematical approximation
        approx_matrix = np.dot(Uk, np.dot(Sk, Vtk))

        # 4. Calculate the Residual Matrix (Modulo 256 Ring)
        # This step guarantees 100% Lossless Integrity by capturing the exact floating-point truncation error.
        # Formula: (Original - Approximation) % 256
        residual_matrix = (matrix.astype(int) - np.round(approx_matrix).astype(int)) % 256
        
        # Package the Operator (U, S, V^T) and return with the Residual
        spectral_component = (Uk, S[:self.rank_k], Vtk)
        return spectral_component, residual_matrix.astype(np.uint8)

    def reintegrate(self, spectral_component, residual_matrix):
        """
        The Reintegration process (Data Healing).
        
        :param spectral_component: The structural skeleton (Operator).
        :param residual_matrix: The exact byte differences (The Ash).
        :return: The perfectly reconstructed original data matrix.
        """
        Uk, S_vector, Vtk = spectral_component
        Sk = np.diag(S_vector)

        # 1. Rebuild the approximation matrix from the Operator
        restored_approx = np.dot(Uk, np.dot(Sk, Vtk))

        # 2. Modulo addition with the Residual Matrix
        # Formula: (Residual + Approximation) % 256
        final_matrix = (residual_matrix.astype(int) + np.round(restored_approx).astype(int)) % 256

        return final_matrix.astype(np.uint8)

    def verify_integrity(self, original, reconstructed):
        """
        Verifies the lossless nature of the reconstruction.
        Returns True if the Mean Squared Error (MSE) is exactly 0.0.
        """
        mse = np.mean((original - reconstructed) ** 2)
        return mse == 0.0, mse