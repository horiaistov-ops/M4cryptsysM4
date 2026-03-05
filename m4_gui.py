import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import M4 Architecture modules
from m4_core import M4Core
from m4_dynamics import M4Dynamics

class M4App(tk.Tk):
    """
    M4App: The Graphical User Interface (GUI) for the M4 PoC.
    Provides a simple dashboard to load data and a comprehensive 
    Matplotlib scientific view to analyze entropy and structural integrity.
    """
    def __init__(self):
        super().__init__()
        self.title("M4-Architecture: Scientific Prototype (PoC)")
        self.geometry("600x400") # Main menu window; the scientific report opens in a separate figure
        
        # Initialize Core and Dynamics modules
        self.core = M4Core(rank_k=30)
        self.dynamics = M4Dynamics(master_seed="MENDELU_2026_FINAL")
        self.current_seed = self.dynamics.generate_guest_seed()

        style = ttk.Style()
        style.theme_use('clam')

        # Build the Control Dashboard
        main_frame = tk.Frame(self)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        tk.Label(main_frame, text="M4 Security Dashboard", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Action Buttons Frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="🖼️ Load Image & Analyze", 
                  command=self.run_image_analysis, 
                  bg="#4a90e2", fg="white", font=("Arial", 12), width=30, height=2).pack(pady=5)

        tk.Button(btn_frame, text="🎲 Generate Pattern & Analyze", 
                  command=self.run_gen_analysis, 
                  bg="#50e3c2", fg="black", font=("Arial", 12), width=30, height=2).pack(pady=5)

        tk.Label(main_frame, text="* Results will open in a separate Matplotlib scientific window", fg="gray").pack(side=tk.BOTTOM)

    def show_scientific_report(self, original, residual, restored, title="M4 Analysis"):
        """
        Generates a professional scientific report displaying the original data, 
        the residual ash, and the lossless reconstruction, along with an entropy histogram.
        """
        # Create the Matplotlib figure
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # --- ROW 1: VISUAL MATRICES ---
        
        # 1. Original Data
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
        ax1.set_title("1. Original Input")
        ax1.axis('off')

        # 2. Residual (The Ash / Noise)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(residual, cmap='gray', vmin=0, vmax=255)
        ax2.set_title("2. M4 Residual (Ash)\nMaximum Entropy State")
        ax2.axis('off')

        # 3. Reintegrated Data
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(restored, cmap='gray', vmin=0, vmax=255)
        
        # Verify Mean Squared Error (MSE) for mathematical proof of lossless integrity
        mse = np.mean((original - restored) ** 2)
        status = "LOSSLESS" if mse == 0 else f"ERR: {mse:.2f}"
        color = "green" if mse == 0 else "red"
        
        ax3.set_title(f"3. Reintegrated Data\nStatus: {status}", color=color, fontweight='bold')
        ax3.axis('off')

        # --- ROW 2: ENTROPY HISTOGRAM ---
        
        ax_hist = fig.add_subplot(2, 1, 2) # Single wide subplot for histogram
        
        # Flatten matrices for histogram distribution analysis
        orig_flat = original.flatten()
        res_flat = residual.flatten()
        
        # Plot histograms to visually prove the destruction of structure
        ax_hist.hist(orig_flat, bins=256, range=(0, 255), color='blue', alpha=0.5, label='Original Structure (High Info)')
        ax_hist.hist(res_flat, bins=256, range=(0, 255), color='red', alpha=0.5, label='M4 Ash (Max Entropy / White Noise)')
        
        ax_hist.set_title("Differential Entropy Analysis")
        ax_hist.set_xlim([0, 255])
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Render the window
        plt.tight_layout()
        plt.show()

    def run_image_analysis(self):
        """Pipeline for executing M4 Architecture on an external image file."""
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not file_path: return

        original_matrix = self.core.prepare_image(file_path)
        
        # 0. SVD Bifurcation
        spectral, residual = self.core.bifurcate(original_matrix)
        
        # 1. Generate a unique Cryptographic Nonce (IV) for this specific file
        file_nonce = self.dynamics.generate_file_nonce()
        
        # 2. Apply dynamic spatial dislocation (Scrambling)
        scrambled_op = self.dynamics.scramble_operator(spectral, self.current_seed, file_nonce)
        final_res = self.dynamics.scramble_residual(residual, self.current_seed, file_nonce)
        
        # 3. Reverse the dislocation (Descrambling) using the exact same Nonce
        rest_op = self.dynamics.descramble_operator(scrambled_op, self.current_seed, file_nonce)
        rest_res = self.dynamics.descramble_residual(final_res, self.current_seed, file_nonce)
        
        # 4. Final Modulo Reintegration
        final_matrix = self.core.reintegrate(rest_op, rest_res)
        
        file_name = file_path.split('/')[-1]
        self.show_scientific_report(original_matrix, final_res, final_matrix, title=f"M4 Analysis: {file_name}")

    def run_gen_analysis(self):
        """Pipeline for executing M4 Architecture on a mathematically generated pattern."""
        # Generate a mathematical plasma pattern for testing
        x = np.linspace(0, 6 * np.pi, 256)
        y = np.linspace(0, 6 * np.pi, 256)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.random.uniform(0.5, 2)*X) + np.cos(np.random.uniform(0.5, 2)*Y) + np.sin((X*Y)/10)
        gen_matrix = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)

        # 0. SVD Bifurcation
        spectral, residual = self.core.bifurcate(gen_matrix)
        
        # 1. Generate a unique Cryptographic Nonce (IV) for the generated pattern
        file_nonce = self.dynamics.generate_file_nonce()
        
        # 2. Scramble
        scrambled_op = self.dynamics.scramble_operator(spectral, self.current_seed, file_nonce)
        final_res = self.dynamics.scramble_residual(residual, self.current_seed, file_nonce)
        
        # 3. Descramble
        rest_op = self.dynamics.descramble_operator(scrambled_op, self.current_seed, file_nonce)
        rest_res = self.dynamics.descramble_residual(final_res, self.current_seed, file_nonce)
        
        # 4. Reintegrate
        final_matrix = self.core.reintegrate(rest_op, rest_res)

        self.show_scientific_report(gen_matrix, final_res, final_matrix, title="M4 Analysis: Generated Plasma Pattern")

if __name__ == "__main__":
    app = M4App()
    app.mainloop()