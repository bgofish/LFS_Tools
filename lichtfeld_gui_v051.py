import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import subprocess
import platform
import re
import json
from pathlib import Path

class LichtFeldStudioDialog:
    def __init__(self, root):
        self.root = root
        self.root.title("LichtFeld Studio CLI Batch File Generator v0.51 - 3D Gaussian Splatting")
        self.root.geometry("985x1100")
        
        # Variables to store all options
        self.master_folder_path = tk.StringVar(value="U:/CLI")
        self.folder_path = tk.StringVar()
        self.option_vars = []
        self.option_entries = []
        self.option_checkbuttons = []
        self.option_states = []
        
        # Frame processing variables
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.start_frame_var = tk.StringVar(value="001")
        self.end_frame_var = tk.StringVar(value="010")
        self.nth_frame_var = tk.StringVar(value="1")
        
        # Training control variables
        self.base_output_name = tk.StringVar(value="")
        
        # Export path options
        self.combined_export_path_var = tk.BooleanVar(value=True)
        
        # Frame ID for filename option
        self.use_frame_id_var = tk.BooleanVar(value=True)
        
        # Undistorted U_Folder option
        self.use_undistorted_var = tk.BooleanVar(value=False)
        
        # Expert mode option (suppresses confirmations)
        self.expert_mode_var = tk.BooleanVar(value=False)
        
        # Executable path variable
        self.executable_path = tk.StringVar(value="U:/LFS/LichtFeld-Studio/build/LichtFeld-Studio.exe")
        
        # New options for LFS3
        self.project_path_var = tk.StringVar(value="")
        self.view_ply_var = tk.StringVar(value="")
        self.view_mode_var = tk.BooleanVar(value=False)
        self.resume_checkpoint_var = tk.StringVar(value="")
        
        # Progress tracking
        self.progress_log = []
        
        # Trace changes to master folder to update paths
        self.master_folder_path.trace('w', self.update_master_folder_paths)
        
        # Set up the main frame with scrolling
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize paths
        self.update_master_folder_paths()
        
        # Create a canvas for scrolling
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create the executable path section (at the very top)
        self.create_executable_section()
        
        # Create the project path and view mode section
        self.create_project_section()
        
        # Create the folder selection section with buttons
        self.create_folder_section()
        
        # Create the special modes section
        self.create_special_modes_section()
        
        # Create the options section (organized by category)
        self.create_options_section()
        
        # Create the output section
        self.create_output_section()
    
    def create_executable_section(self):
        """Create the executable path selection section at the top"""
        exec_frame = ttk.LabelFrame(self.scrollable_frame, text="LichtFeld Studio Executable Path")
        exec_frame.pack(fill="x", padx=5, pady=5)
        
        # Executable path selection
        ttk.Label(exec_frame, text="Executable Path:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(exec_frame, textvariable=self.executable_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(exec_frame, text="Browse...", command=self.browse_executable).grid(row=0, column=2, padx=5, pady=5)
        
        # Help text
        ttk.Label(exec_frame, text="Specify the full path to LichtFeld-Studio.exe (supports paths with spaces)", 
                 font=("", 8), foreground="gray").grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=(0,5))
    
    def create_project_section(self):
        """Create the project path and view mode selection section"""
        project_frame = ttk.LabelFrame(self.scrollable_frame, text="Mode Selection")
        project_frame.pack(fill="x", padx=5, pady=5)
        
        # Mode selection
        mode_frame = ttk.Frame(project_frame)
        mode_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Training Mode", variable=self.view_mode_var, 
                       value=False, command=self.toggle_view_mode).pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="View Mode", variable=self.view_mode_var, 
                       value=True, command=self.toggle_view_mode).pack(side="left", padx=15)
        
        # Config file selection (for training mode)
        self.config_file_frame = ttk.Frame(project_frame)
        self.config_file_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(self.config_file_frame, text="Config File (optional):").pack(side="left", padx=(0,5))
        ttk.Entry(self.config_file_frame, textvariable=self.project_path_var, width=50).pack(side="left", padx=5)
        ttk.Button(self.config_file_frame, text="Browse...", command=self.browse_config_file).pack(side="left", padx=5)
        
        # Resume checkpoint selection (for training mode)
        self.resume_frame = ttk.Frame(project_frame)
        self.resume_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(self.resume_frame, text="Resume Checkpoint:").pack(side="left", padx=(0,5))
        ttk.Entry(self.resume_frame, textvariable=self.resume_checkpoint_var, width=50).pack(side="left", padx=5)
        ttk.Button(self.resume_frame, text="Browse...", command=self.browse_resume_checkpoint).pack(side="left", padx=5)
        
        # View file selection (for view mode)
        self.view_ply_frame = ttk.Frame(project_frame)
        self.view_ply_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(self.view_ply_frame, text="View File/Directory:").pack(side="left", padx=(0,5))
        ttk.Entry(self.view_ply_frame, textvariable=self.view_ply_var, width=50).pack(side="left", padx=5)
        ttk.Button(self.view_ply_frame, text="Browse File...", command=self.browse_view_file).pack(side="left", padx=2)
        ttk.Button(self.view_ply_frame, text="Browse Dir...", command=self.browse_view_directory).pack(side="left", padx=2)
        
        # Initially show training mode
        self.toggle_view_mode()
        
    def create_folder_section(self):
        # Main container frame for folder selection and buttons
        main_container = ttk.Frame(self.scrollable_frame)
        main_container.pack(fill="x", padx=5, pady=5)
        
        # Left side: Input Folder Selection
        folder_frame = ttk.LabelFrame(main_container, text="Dataset Folder Selection")
        folder_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Master Folder
        ttk.Label(folder_frame, text="Master Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.master_folder_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(folder_frame, text="Browse...", command=self.browse_master_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # Processing Mode selection
        mode_frame = ttk.LabelFrame(folder_frame, text="Processing Mode", padding="5")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Single Dataset Processing", variable=self.batch_mode_var, 
                       value=False, command=self.toggle_batch_mode).pack(anchor="w")
        
        batch_radio_frame = ttk.Frame(mode_frame)
        batch_radio_frame.pack(fill="x", pady=2)
        
        ttk.Radiobutton(batch_radio_frame, text="Batch Processing (FRAME", variable=self.batch_mode_var, 
                       value=True, command=self.toggle_batch_mode).pack(side="left")
        
        ttk.Entry(batch_radio_frame, textvariable=self.start_frame_var, width=5).pack(side="left", padx=2)
        ttk.Label(batch_radio_frame, text="to").pack(side="left", padx=2)
        ttk.Entry(batch_radio_frame, textvariable=self.end_frame_var, width=5).pack(side="left", padx=2)
        ttk.Label(batch_radio_frame, text=")").pack(side="left")
        ttk.Label(batch_radio_frame, text="Every n-th:").pack(side="left", padx=(10,2))
        ttk.Entry(batch_radio_frame, textvariable=self.nth_frame_var, width=5).pack(side="left", padx=(0,2))
        
        # Use Undistorted U_Folder checkbox
        undistorted_frame = ttk.Frame(mode_frame)
        undistorted_frame.pack(fill="x", pady=2)
        
        ttk.Checkbutton(undistorted_frame, text="Use Undistorted U_Folder (Frame### → UFrame###)", 
                       variable=self.use_undistorted_var, command=self.toggle_undistorted_mode).pack(anchor="w")
        
        # Input dataset folder (now depends on mode)
        ttk.Label(folder_frame, text="Dataset Folder:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_path, width=40)
        self.folder_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(folder_frame, text="Browse...", command=self.browse_folder).grid(row=2, column=2, padx=5, pady=5)
        
        # Right side: Action Buttons
        self.create_action_buttons(main_container)
    
    def create_special_modes_section(self):
        """Create special modes section for boolean flags"""
        modes_frame = ttk.LabelFrame(self.scrollable_frame, text="Special Modes")
        modes_frame.pack(fill="x", padx=5, pady=5)
        
        # Special boolean flags
        self.eval_var = tk.BooleanVar(value=False)
        self.headless_var = tk.BooleanVar(value=True)
        self.train_var = tk.BooleanVar(value=False)
        self.enable_mip_var = tk.BooleanVar(value=False)
        self.bilateral_grid_var = tk.BooleanVar(value=False)
        self.save_eval_images_var = tk.BooleanVar(value=False)
        self.save_depth_var = tk.BooleanVar(value=False)
        self.random_var = tk.BooleanVar(value=False)
        self.gut_var = tk.BooleanVar(value=False)
        self.enable_sparsity_var = tk.BooleanVar(value=False)
        self.bg_modulation_var = tk.BooleanVar(value=False)
        self.ppisp_var = tk.BooleanVar(value=False)
        self.ppisp_controller_var = tk.BooleanVar(value=False)
        self.no_interop_var = tk.BooleanVar(value=False)
        self.invert_masks_var = tk.BooleanVar(value=False)
        self.no_alpha_as_mask_var = tk.BooleanVar(value=False)
        self.no_cpu_cache_var = tk.BooleanVar(value=False)
        self.no_fs_cache_var = tk.BooleanVar(value=False)
        self.use_error_map_var = tk.BooleanVar(value=False)
        self.use_edge_map_var = tk.BooleanVar(value=False)
        self.ppisp_freeze_var = tk.BooleanVar(value=False)
        self.debug_python_var = tk.BooleanVar(value=False)
        
        # First row
        row1_frame = ttk.Frame(modes_frame)
        row1_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(row1_frame, text="Enable evaluation (--eval)", 
                       variable=self.eval_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row1_frame, text="Headless mode (--headless)", 
                       variable=self.headless_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row1_frame, text="Auto-start training (--train)", 
                       variable=self.train_var).pack(side="left")
        
        # Second row
        row2_frame = ttk.Frame(modes_frame)
        row2_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(row2_frame, text="Enable mip/anti-aliasing (--enable-mip)", 
                       variable=self.enable_mip_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row2_frame, text="Bilateral grid (--bilateral-grid)", 
                       variable=self.bilateral_grid_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row2_frame, text="PPISP appearance modeling (--ppisp)", 
                       variable=self.ppisp_var).pack(side="left")
        
        # Third row
        row3_frame = ttk.Frame(modes_frame)
        row3_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(row3_frame, text="Save eval images (--save-eval-images)", 
                       variable=self.save_eval_images_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row3_frame, text="Save depth maps (--save-depth)", 
                       variable=self.save_depth_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row3_frame, text="No CUDA-GL interop (--no-interop)", 
                       variable=self.no_interop_var).pack(side="left")
        
        # Fourth row
        row4_frame = ttk.Frame(modes_frame)
        row4_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(row4_frame, text="Random init (--random)", 
                       variable=self.random_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row4_frame, text="GUT mode (--gut)", 
                       variable=self.gut_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row4_frame, text="Enable sparsity (--enable-sparsity)", 
                       variable=self.enable_sparsity_var).pack(side="left")
        
        # Fifth row
        row5_frame = ttk.Frame(modes_frame)
        row5_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(row5_frame, text="BG modulation (--bg-modulation)", 
                       variable=self.bg_modulation_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row5_frame, text="Invert masks (--invert-masks)", 
                       variable=self.invert_masks_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row5_frame, text="PPISP controller novel views (--ppisp-controller)", 
                       variable=self.ppisp_controller_var).pack(side="left")

        # Sixth row
        row6_frame = ttk.Frame(modes_frame)
        row6_frame.pack(fill="x", padx=5, pady=2)

        ttk.Checkbutton(row6_frame, text="No alpha-as-mask for RGBA (--no-alpha-as-mask)", 
                       variable=self.no_alpha_as_mask_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row6_frame, text="Disable CPU cache (--no-cpu-cache)", 
                       variable=self.no_cpu_cache_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row6_frame, text="Disable filesystem cache (--no-fs-cache)", 
                       variable=self.no_fs_cache_var).pack(side="left")

        # Seventh row - new V051 options
        row7_frame = ttk.Frame(modes_frame)
        row7_frame.pack(fill="x", padx=5, pady=2)

        ttk.Checkbutton(row7_frame, text="Use error map for MRNF (--use-error-map)",
                       variable=self.use_error_map_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row7_frame, text="Use edge map for MRNF (--use-edge-map)",
                       variable=self.use_edge_map_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row7_frame, text="Freeze PPISP learning (--ppisp-freeze)",
                       variable=self.ppisp_freeze_var).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(row7_frame, text="Debug Python plugin (--debug-python)",
                       variable=self.debug_python_var).pack(side="left")

    def create_options_section(self):
        # Create notebook for organized option categories
        notebook = ttk.Notebook(self.scrollable_frame)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Training Options Tab
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Training Options")
        self.create_training_options(training_frame)
        
        # Rendering Options Tab
        rendering_frame = ttk.Frame(notebook)
        notebook.add(rendering_frame, text="Rendering Options")
        self.create_rendering_options(rendering_frame)
        
        # Optimization Options Tab
        optimization_frame = ttk.Frame(notebook)
        notebook.add(optimization_frame, text="Optimization Options")
        self.create_optimization_options(optimization_frame)
        
        # Sparsity & Memory Tab
        sparsity_frame = ttk.Frame(notebook)
        notebook.add(sparsity_frame, text="Sparsity & Memory")
        self.create_sparsity_options(sparsity_frame)
        
        # Logging & Masking Tab
        logging_frame = ttk.Frame(notebook)
        notebook.add(logging_frame, text="Logging & Masking")
        self.create_logging_options(logging_frame)
    
    def create_training_options(self, parent):
        # Training options based on LichtFeld Studio help
        training_options = [
            {"name": "--data-path", "default": "", "description": "Path to training data", "enabled": True},
            {"name": "--output-path", "default": "./output", "description": "Path to output", "enabled": True},
            {"name": "--iter", "default": "30000", "description": "Number of iterations", "enabled": True},
            {"name": "--max-cap", "default": "", "description": "Max Gaussians for MCMC", "enabled": False},
            {"name": "--images", "default": "images", "description": "Images folder name", "enabled": False},
            {"name": "--test-every", "default": "", "description": "Use every Nth image as test", "enabled": False},
            {"name": "--steps-scaler", "default": "", "description": "Scale training steps by factor", "enabled": False},
            {"name": "--init-num-pts", "default": "", "description": "Number of random initialization points", "enabled": False},
            {"name": "--init-extent", "default": "", "description": "Extent of random initialization", "enabled": False},
            {"name": "--init", "default": "", "description": "Initialize from splat file (.ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume)", "enabled": False},
            {"name": "--max-width", "default": "3840", "description": "Max width of images in px", "enabled": False},
            {"name": "--resize_factor", "default": "auto", "description": "Resize factor: auto, 1, 2, 4, 8", "enabled": False}
        ]
        
        self.create_option_widgets(parent, training_options, 0)
    
    def create_rendering_options(self, parent):
        # Rendering options based on LichtFeld Studio help
        rendering_options = [
            {"name": "--sh-degree", "default": "3", "description": "Max SH degree [0-3]", "enabled": False},
            {"name": "--sh-degree-interval", "default": "", "description": "SH degree interval", "enabled": False},
            {"name": "--min-opacity", "default": "", "description": "Minimum opacity threshold", "enabled": False},
            {"name": "--import-cameras", "default": "", "description": "Import COLMAP cameras from sparse folder (no images required)", "enabled": False},
            {"name": "--timelapse-images", "default": "", "description": "Image filenames for timelapse (space-separated)", "enabled": False},
            {"name": "--timelapse-every", "default": "50", "description": "Render timelapse every N iterations", "enabled": False},
            {"name": "--ppisp-sidecar", "default": "", "description": "Path to PPISP sidecar (.ppisp) for frozen PPISP training", "enabled": False}
        ]
        
        start_index = len(self.option_vars)
        self.create_option_widgets(parent, rendering_options, start_index)
    
    def create_optimization_options(self, parent):
        # Optimization options based on LichtFeld Studio help
        optimization_options = [
            {"name": "--strategy", "default": "mcmc", "description": "Optimization strategy: mcmc, mrnf, igs+ (legacy aliases: mnrf, lfs)", "enabled": False}
        ]
        
        start_index = len(self.option_vars)
        self.create_option_widgets(parent, optimization_options, start_index)
    
    def create_sparsity_options(self, parent):
        # Sparsity and memory options
        sparsity_options = [
            {"name": "--tile-mode", "default": "1", "description": "Tile mode: 1, 2, 4 tiles for memory efficiency", "enabled": False},
            {"name": "--sparsify-steps", "default": "15000", "description": "Number of steps for sparsification", "enabled": False},
            {"name": "--init-rho", "default": "0.0005", "description": "Initial ADMM penalty parameter", "enabled": False},
            {"name": "--prune-ratio", "default": "0.6", "description": "Final pruning ratio for sparsity", "enabled": False}
        ]
        
        start_index = len(self.option_vars)
        self.create_option_widgets(parent, sparsity_options, start_index)
    
    def create_logging_options(self, parent):
        # Logging and masking options
        logging_options = [
            {"name": "--log-level", "default": "info", "description": "Log level: trace, debug, info, perf, warn, error, critical, off", "enabled": False},
            {"name": "--log-file", "default": "", "description": "Optional log file path", "enabled": False},
            {"name": "--log-filter", "default": "", "description": "Filter log messages (glob: *foo*, regex: \\d+)", "enabled": False},
            {"name": "--mask-mode", "default": "none", "description": "Mask mode: none, segment, ignore, alpha_consistent", "enabled": False},
            {"name": "--debug-python-port", "default": "5678", "description": "Port for debugpy listener (requires --debug-python flag)", "enabled": False}
        ]
        
        start_index = len(self.option_vars)
        self.create_option_widgets(parent, logging_options, start_index)
    
    def create_option_widgets(self, parent, options, start_index):
        """Create widgets for a set of options"""
        for i, option in enumerate(options):
            var = tk.StringVar(value=option["default"])
            state_var = tk.BooleanVar(value=option["enabled"])
            
            self.option_vars.append(var)
            self.option_states.append(state_var)
            
            # Option name
            ttk.Label(parent, text=f"{start_index + i + 1}. {option['name']}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            # Option entry
            entry = ttk.Entry(parent, textvariable=var, width=30)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.option_entries.append(entry)
            
            # Include checkbox
            cb = ttk.Checkbutton(parent, text="Include", variable=state_var)
            cb.grid(row=i, column=2, padx=5, pady=2)
            self.option_checkbuttons.append(cb)
            
            # Special handling for --output-path option
            if option["name"] == "--output-path":
                # Add Frame ID radio button option
                frame_id_frame = ttk.Frame(parent)
                frame_id_frame.grid(row=i, column=4, sticky="w", padx=5, pady=2)
                
                ttk.Radiobutton(frame_id_frame, text="Use Frame ID for output path", 
                               variable=self.use_frame_id_var, value=False,
                               command=self.toggle_frame_id_mode).pack(anchor="w")
                ttk.Radiobutton(frame_id_frame, text="Use path entry", 
                               variable=self.use_frame_id_var, value=True,
                               command=self.toggle_frame_id_mode).pack(anchor="w")
                
                # Description with extended info
                desc_text = option["description"] + " | Toggle Frame ID mode above"
                ttk.Label(parent, text=desc_text, font=("", 8)).grid(row=i, column=3, sticky="w", padx=5, pady=2)
            else:
                # Description for regular options
                ttk.Label(parent, text=option["description"], font=("", 8)).grid(row=i, column=3, sticky="w", padx=5, pady=2)
    
    def create_output_section(self):
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Generated Command Preview & Progress Log")
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Command output
        ttk.Label(output_frame, text="Complete Command Preview:").pack(anchor="w", padx=5, pady=(5,0))
        self.output_text = ScrolledText(output_frame, width=80, height=12, wrap=tk.NONE)
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Progress log
        ttk.Label(output_frame, text="Generation Log:").pack(anchor="w", padx=5, pady=(10,0))
        self.log_text = ScrolledText(output_frame, width=80, height=4)
        self.log_text.pack(fill="x", padx=5, pady=5)
    
    def create_action_buttons(self, parent_frame):
        # Create a vertical container for button groups on the right side
        button_container = ttk.Frame(parent_frame)
        button_container.pack(side="right", fill="y", padx=(5, 0))
        
        # Action Buttons
        action_frame = ttk.LabelFrame(button_container, text="Action Buttons", padding="5")
        action_frame.pack(fill="x")
        
        # Top row of action buttons
        top_action_frame = ttk.Frame(action_frame)
        top_action_frame.pack(fill="x", pady=(0, 3))
        
        ttk.Button(top_action_frame, text="🔄 Generate Command", command=self.generate_command, 
                  style="Accent.TButton").pack(side="left", padx=2)
        ttk.Button(top_action_frame, text="💾 Save Batch File", command=self.save_batch, 
                  style="Accent.TButton").pack(side="left", padx=2)
        ttk.Button(top_action_frame, text="▶️ Execute", command=self.execute_command).pack(side="left", padx=2)
        ttk.Button(top_action_frame, text="Exit", command=self.root.quit).pack(side="right", padx=2)
        
        # Bottom row of action buttons
        bottom_action_frame = ttk.Frame(action_frame)
        bottom_action_frame.pack(fill="x")
        
        ttk.Button(bottom_action_frame, text="📋 Copy to Clipboard", command=self.copy_to_clipboard).pack(side="left", padx=2)
        ttk.Button(bottom_action_frame, text="💾 Save Config", command=self.save_config).pack(side="left", padx=2)
        ttk.Button(bottom_action_frame, text="📂 Load Config", command=self.load_config).pack(side="left", padx=2)
        ttk.Button(bottom_action_frame, text="Reset All", command=self.reset_all).pack(side="right", padx=2)
        
        # Third row for batch processing
        batch_action_frame = ttk.Frame(action_frame)
        batch_action_frame.pack(fill="x", pady=(3, 0))
        
        ttk.Button(batch_action_frame, text="Generate Batch Files for All Frames & save in [Batch] folder", 
                  command=self.generate_all_frames_batch).pack(side="left", padx=2)
        
        # Expert Mode toggle
        expert_mode_frame = ttk.Frame(action_frame)
        expert_mode_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Checkbutton(expert_mode_frame, text="Expert Mode - Suppress all confirmation dialogs", 
                       variable=self.expert_mode_var).pack(anchor="w", padx=5)
    
    def update_master_folder_paths(self, *args):
        """Update paths when master folder changes."""
        master_dir = self.master_folder_path.get()
        if master_dir:
            # Determine frame prefix based on undistorted mode
            frame_prefix = "UFrame" if self.use_undistorted_var.get() else "Frame"
            
            # Update input folder path based on batch mode
            if not self.batch_mode_var.get():
                # Single folder mode - use master folder directly
                self.folder_path.set(master_dir.replace("\\", "/"))
            else:
                # Batch mode - show template
                self.folder_path.set(os.path.join(master_dir, f"{frame_prefix}###").replace("\\", "/"))
    
    def toggle_batch_mode(self):
        """Toggle UI elements based on batch processing mode."""
        # Determine frame prefix based on undistorted mode
        frame_prefix = "UFrame" if self.use_undistorted_var.get() else "Frame"
        
        if self.batch_mode_var.get():
            # Batch mode - disable individual folder selection
            self.folder_entry.config(state=tk.DISABLED)
            master_dir = self.master_folder_path.get() or "U:/CLI"
            self.folder_path.set(os.path.join(master_dir, f"{frame_prefix}###").replace("\\", "/"))
        else:
            # Single mode - enable individual folder selection
            self.folder_entry.config(state=tk.NORMAL)
            master_dir = self.master_folder_path.get() or "U:/CLI"
            # Use master folder directly for single dataset processing
            self.folder_path.set(master_dir.replace("\\", "/"))
    
    def toggle_frame_id_mode(self):
        """Toggle Frame ID mode for --output-path option"""
        pass
    
    def toggle_undistorted_mode(self):
        """Toggle between Frame### and UFrame### when undistorted mode changes"""
        self.update_master_folder_paths()
    
    def toggle_view_mode(self):
        """Toggle between training mode and view mode"""
        if self.view_mode_var.get():
            # View mode - show view file selection, hide training options
            self.config_file_frame.pack_forget()
            self.resume_frame.pack_forget()
            self.view_ply_frame.pack(fill="x", padx=5, pady=2)
        else:
            # Training mode - show training options, hide view selection
            self.view_ply_frame.pack_forget()
            self.config_file_frame.pack(fill="x", padx=5, pady=2)
            self.resume_frame.pack(fill="x", padx=5, pady=2)
    
    def browse_config_file(self):
        """Browse for LichtFeld Studio config file (JSON)"""
        file_selected = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="U:/CLI"
        )
        if file_selected:
            self.project_path_var.set(file_selected.replace("\\", "/"))
    
    def browse_resume_checkpoint(self):
        """Browse for checkpoint file to resume from"""
        file_selected = filedialog.askopenfilename(
            title="Select Resume Checkpoint",
            filetypes=[("Resume files", "*.resume"), ("All files", "*.*")],
            initialdir="U:/CLI"
        )
        if file_selected:
            self.resume_checkpoint_var.set(file_selected.replace("\\", "/"))
    
    def browse_view_file(self):
        """Browse for splat file to view"""
        file_selected = filedialog.askopenfilename(
            title="Select File to View",
            filetypes=[
                ("Splat files", "*.ply *.sog *.spz *.usd *.usda *.usdc *.usdz"),
                ("Mesh files", "*.obj *.fbx *.gltf *.glb *.stl"),
                ("PLY files", "*.ply"),
                ("SOG files", "*.sog"),
                ("SPZ files", "*.spz"),
                ("USD files", "*.usd *.usda *.usdc *.usdz"),
                ("OBJ files", "*.obj"),
                ("FBX files", "*.fbx"),
                ("GLTF files", "*.gltf *.glb"),
                ("STL files", "*.stl"),
                ("All files", "*.*")
            ],
            initialdir="U:/CLI"
        )
        if file_selected:
            self.view_ply_var.set(file_selected.replace("\\", "/"))
    
    def browse_view_directory(self):
        """Browse for directory containing splat files"""
        folder_selected = filedialog.askdirectory(title="Select Directory with Splat Files")
        if folder_selected:
            self.view_ply_var.set(folder_selected.replace("\\", "/"))
    
    def browse_master_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Master Folder")
        if folder_selected:
            self.master_folder_path.set(folder_selected.replace("\\", "/"))
    
    def browse_folder(self):
        master_dir = self.master_folder_path.get()
        initial_dir = master_dir if master_dir and os.path.exists(master_dir) else None
        
        folder_selected = filedialog.askdirectory(
            title="Select Dataset Folder",
            initialdir=initial_dir
        )
        if folder_selected:
            self.folder_path.set(folder_selected.replace("\\", "/"))
    
    def browse_executable(self):
        """Browse for the LichtFeld-Studio.exe executable"""
        file_selected = filedialog.askopenfilename(
            title="Select LichtFeld Studio Executable",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")],
            initialdir="C:/"
        )
        if file_selected:
            self.executable_path.set(file_selected.replace("\\", "/"))
    
    def get_frame_code(self):
        """Extract frame code (e.g., F001 or UF001) from the folder_path."""
        folder = self.folder_path.get().strip()
        
        # Check for UFrame### pattern first
        m = re.search(r"UFrame(\d{3})", folder, re.IGNORECASE)
        if m:
            return f"UF{m.group(1)}"
        
        # Check for Frame### pattern
        m = re.search(r"Frame(\d{3})", folder, re.IGNORECASE)
        if m:
            return f"F{m.group(1)}"
        
        return ""
    
    def get_export_paths_to_create(self, use_full_paths=False):
        """Get list of export paths that need to be created for current settings"""
        export_paths = []
        
        # Always get the output path from the --output-path option when it's enabled
        output_path_value = None
        for j, var in enumerate(self.option_vars):
            if self.get_option_name_by_index(j) == "--output-path" and self.option_states[j].get():
                output_path_value = var.get().strip()
                break
        
        if output_path_value:
            export_paths.append(output_path_value)
        else:
            dataset_path = self.folder_path.get().strip()
            if use_full_paths and dataset_path:
                base_output_path = os.path.join(dataset_path, "output")
                export_paths.append(base_output_path)
            else:
                export_paths.append("output")
        
        return export_paths

# LichtFeld Studio GUI v0.5
# This continues from Part 2 - append this after Part 2

    def generate_command(self):
        """Generate LichtFeld Studio command"""
        # Start with the custom executable path (with proper quoting for spaces)
        exe_path = self.executable_path.get().strip()
        if not exe_path:
            exe_path = "LichtFeld-Studio.exe"
        
        # Quote the executable path if it contains spaces
        if " " in exe_path and not exe_path.startswith('"'):
            exe_path = f'"{exe_path}"'
        
        command_parts = [exe_path]
        
        # Handle view mode first (mutually exclusive with training)
        if self.view_mode_var.get():
            view_file = self.view_ply_var.get().strip()
            if not view_file:
                messagebox.showerror("Error", "Please select a file or directory to view!")
                return
            
            # Quote the view file path if it contains spaces
            if " " in view_file and not view_file.startswith('"'):
                view_file = f'"{view_file}"'
            command_parts.append(f"--view={view_file}")
            
            # For view mode, we don't need training options
            command_str = " ".join(command_parts)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, command_str)
            self.log_message(f"Generated view command: {view_file}")
            return command_str
        
        # Handle resume checkpoint (mutually exclusive with normal training start)
        resume_checkpoint = self.resume_checkpoint_var.get().strip()
        if resume_checkpoint:
            if " " in resume_checkpoint and not resume_checkpoint.startswith('"'):
                resume_checkpoint = f'"{resume_checkpoint}"'
            command_parts.append(f"--resume={resume_checkpoint}")
            self.log_message(f"Resume mode: {resume_checkpoint}")
        
        # Handle config file for training mode
        config_path = self.project_path_var.get().strip()
        if config_path:
            if " " in config_path and not config_path.startswith('"'):
                config_path = f'"{config_path}"'
            command_parts.append(f"--config={config_path}")
        
        # Add special mode flags (boolean options)
        if self.eval_var.get():
            command_parts.append("--eval")
        if self.headless_var.get():
            command_parts.append("--headless")
        if self.train_var.get():
            command_parts.append("--train")
        if self.enable_mip_var.get():
            command_parts.append("--enable-mip")
        if self.bilateral_grid_var.get():
            command_parts.append("--bilateral-grid")
        if self.save_eval_images_var.get():
            command_parts.append("--save-eval-images")
        if self.save_depth_var.get():
            command_parts.append("--save-depth")
        if self.bg_modulation_var.get():
            command_parts.append("--bg-modulation")
        if self.ppisp_var.get():
            command_parts.append("--ppisp")
        if self.ppisp_controller_var.get():
            command_parts.append("--ppisp-controller")
        if self.random_var.get():
            command_parts.append("--random")
        if self.gut_var.get():
            command_parts.append("--gut")
        if self.enable_sparsity_var.get():
            command_parts.append("--enable-sparsity")
        if self.no_interop_var.get():
            command_parts.append("--no-interop")
        if self.invert_masks_var.get():
            command_parts.append("--invert-masks")
        if self.no_alpha_as_mask_var.get():
            command_parts.append("--no-alpha-as-mask")
        if self.no_cpu_cache_var.get():
            command_parts.append("--no-cpu-cache")
        if self.no_fs_cache_var.get():
            command_parts.append("--no-fs-cache")
        if self.use_error_map_var.get():
            command_parts.append("--use-error-map")
        if self.use_edge_map_var.get():
            command_parts.append("--use-edge-map")
        if self.ppisp_freeze_var.get():
            command_parts.append("--ppisp-freeze")
        if self.debug_python_var.get():
            command_parts.append("--debug-python")
        
        # Add all enabled options
        for i, (var, state_var) in enumerate(zip(self.option_vars, self.option_states)):
            if state_var.get():
                value = var.get().strip()
                option_name = self.get_option_name_by_index(i)
                
                # Handle special cases
                if option_name == "--data-path":
                    dataset_path = self.folder_path.get().strip()
                    if dataset_path:
                        if " " in dataset_path and not dataset_path.startswith('"'):
                            dataset_path = f'"{dataset_path}"'
                        command_parts.append(f"{option_name}={dataset_path}")
                
                elif option_name == "--output-path":
                    # Check if we should use Frame ID for output path
                    if self.use_frame_id_var.get():
                        frame_code = self.get_frame_code()
                        dataset_path = self.folder_path.get().strip()
                        
                        if self.combined_export_path_var.get():
                            master_dir = self.master_folder_path.get().strip()
                            if master_dir:
                                full_output_path = os.path.join(master_dir, "output", frame_code).replace('\\', '/')
                                value = full_output_path
                            else:
                                value = f"./output/{frame_code}"
                        else:
                            if dataset_path:
                                full_output_path = os.path.join(dataset_path, "output").replace('\\', '/')
                                value = full_output_path
                            else:
                                value = "./output"
                        
                        # Create folder if it doesn't exist
                        try:
                            if not os.path.exists(value):
                                os.makedirs(value)
                                self.log_message(f"Created output folder: {value}")
                        except Exception as e:
                            self.log_message(f"Warning: Could not create output folder: {str(e)}")
                    
                    if " " in value and not value.startswith('"'):
                        value = f'"{value}"'
                    command_parts.append(f"{option_name}={value}")
                
                else:
                    # Regular options with values
                    if value:
                        if " " in value and not value.startswith('"'):
                            value = f'"{value}"'
                        command_parts.append(f"{option_name}={value}")
        
        # Join command
        command_str = " ".join(command_parts)
        
        # Show in the output area
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, command_str)
        
        # Log the generation
        dataset_path = self.folder_path.get().strip()
        self.log_message(f"Generated command for dataset: {dataset_path}")
        
        return command_str
    
    def get_option_name_by_index(self, index):
        """Get the option name by its index in the combined list"""
        all_option_names = [
            # Training options (0-11)
            "--data-path", "--output-path", "--iter", "--max-cap", "--images",
            "--test-every", "--steps-scaler", "--init-num-pts", "--init-extent", "--init", "--max-width", "--resize_factor",
            # Rendering options (12-18)
            "--sh-degree", "--sh-degree-interval", "--min-opacity",
            "--import-cameras", "--timelapse-images", "--timelapse-every", "--ppisp-sidecar",
            # Optimization options (19)
            "--strategy",
            # Sparsity & Memory options (20-23)
            "--tile-mode", "--sparsify-steps", "--init-rho", "--prune-ratio",
            # Logging & Masking options (24-28)
            "--log-level", "--log-file", "--log-filter", "--mask-mode", "--debug-python-port"
        ]
        
        if index < len(all_option_names):
            return all_option_names[index]
        return ""
    
    def log_message(self, message):
        """Add a message to the progress log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def save_batch(self):
        # First generate the command
        command_str = self.generate_command()
        
        if not command_str.strip():
            messagebox.showerror("Error", "No command generated!")
            return
        
        # Create batch file content
        batch_content = "@echo off\n"
        batch_content += "REM Generated LichtFeld Studio CLI Batch File\n"
        batch_content += "REM 3D Gaussian Splatting CUDA Implementation\n"
        batch_content += "REM\n\n"
        
        # Add output folder creation commands
        dataset_path = self.folder_path.get().strip()
        if dataset_path:
            dataset_path_batch = dataset_path.replace('/', '\\')
            batch_content += f'cd /d "{dataset_path_batch}"\n'
            batch_content += "REM Create output folders if they don't exist\n"
            
            output_paths = self.get_export_paths_to_create(use_full_paths=True)
            for output_path in output_paths:
                batch_content += f'if not exist "{output_path}" mkdir "{output_path}"\n'
            
            batch_content += "\n"
        
        # Add the command
        batch_content += command_str + "\n\n"
        
        batch_content += "if errorlevel 1 (\n"
        batch_content += "    echo ERROR: LichtFeld Studio failed!\n"
        batch_content += "    echo Exiting with error code 1\n"
        batch_content += "    pause\n"
        batch_content += "    exit /b 1\n"
        batch_content += ")\n\n"
        batch_content += "echo.\n"
        batch_content += "echo LichtFeld Studio completed successfully!\n"
        batch_content += "echo.\n"
        batch_content += "pause\n"
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".bat",
            filetypes=[("Batch files", "*.bat"), ("All files", "*.*")],
            title="Save LichtFeld Studio Batch File"
        )
        
        if file_path:
            with open(file_path, "w") as f:
                f.write(batch_content)
            if not self.expert_mode_var.get():
                messagebox.showinfo("Success", f"Batch file saved successfully to:\n{file_path}")
    
    def execute_command(self):
        """Execute the LichtFeld Studio command in a new console"""
        command_str = self.output_text.get(1.0, tk.END).strip()
        if not command_str:
            command_str = self.generate_command()
        
        if not command_str.strip():
            messagebox.showerror("Error", "No command generated!")
            return
        
        try:
            dataset_path = self.folder_path.get().strip()
            if dataset_path and os.path.exists(dataset_path):
                folder_creation_commands = []
                output_paths = self.get_export_paths_to_create(use_full_paths=True)
                for output_path in output_paths:
                    folder_creation_commands.append(f'if not exist "{output_path}" mkdir "{output_path}"')
                
                folder_creation_str = "\n".join(folder_creation_commands)
                
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as temp_file:
                    dataset_path_quoted = dataset_path.replace('/', '\\')
                    batch_content = f"@echo off\n"
                    batch_content += f'cd /d "{dataset_path_quoted}"\n'
                    batch_content += "REM Create output folders if they don't exist\n"
                    batch_content += folder_creation_str + "\n"
                    batch_content += "echo Starting LichtFeld Studio...\n"
                    batch_content += "echo.\n"
                    batch_content += f"echo Command: {command_str}\n"
                    batch_content += "echo.\n"
                    batch_content += command_str + "\n"
                    batch_content += "echo.\n"
                    batch_content += "if errorlevel 1 (\n"
                    batch_content += "    echo ERROR: LichtFeld Studio failed!\n"
                    batch_content += "    echo Press any key to close...\n"
                    batch_content += "    pause > nul\n"
                    batch_content += "    exit /b 1\n"
                    batch_content += ") else (\n"
                    batch_content += "    echo LichtFeld Studio completed successfully!\n"
                    batch_content += "    echo Press any key to close...\n"
                    batch_content += "    pause > nul\n"
                    batch_content += ")\n"
                    
                    temp_file.write(batch_content)
                    temp_batch_path = temp_file.name
                
                subprocess.Popen([temp_batch_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
                self.log_message(f"Command executed: {command_str[:100]}...")
                if not self.expert_mode_var.get():
                    messagebox.showinfo("Command Executed", "LichtFeld Studio has been started in a new console window!")
            else:
                subprocess.Popen(command_str, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
                self.log_message(f"Command executed directly: {command_str[:100]}...")
                messagebox.showinfo("Command Executed", "LichtFeld Studio has been started!")
        except Exception as e:
            self.log_message(f"Error executing command: {str(e)}")
            messagebox.showerror("Error", f"Failed to execute command: {str(e)}\n\nTry using 'Save Batch File' and running manually.")
    
    def copy_to_clipboard(self):
        command_str = self.generate_command()
        if command_str.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(command_str)
            self.root.update()
            if not self.expert_mode_var.get():
                messagebox.showinfo("Copy Successful", "Command copied to clipboard!")
        else:
            messagebox.showerror("Error", "No command to copy!")
    
    def reset_all(self):
        """Reset all settings to defaults"""
        if not self.load_settings_from_json():
            self.executable_path.set("U:/LFS/LichtFeld-Studio/build/LichtFeld-Studio.exe")
            self.master_folder_path.set("U:/CLI")
            self.base_output_name.set("")
            self.use_frame_id_var.set(True)
            
            self.batch_mode_var.set(False)
            self.toggle_batch_mode()
            
            self.start_frame_var.set("001")
            self.end_frame_var.set("010")
            self.nth_frame_var.set("1")
            
            self.eval_var.set(False)
            self.headless_var.set(True)
            self.train_var.set(False)
            self.enable_mip_var.set(False)
            self.bilateral_grid_var.set(False)
            self.save_eval_images_var.set(False)
            self.save_depth_var.set(False)
            self.random_var.set(False)
            self.gut_var.set(False)
            self.enable_sparsity_var.set(False)
            self.bg_modulation_var.set(False)
            self.ppisp_var.set(False)
            self.ppisp_controller_var.set(False)
            self.no_interop_var.set(False)
            self.invert_masks_var.set(False)
            self.no_alpha_as_mask_var.set(False)
            self.no_cpu_cache_var.set(False)
            self.no_fs_cache_var.set(False)
            self.use_error_map_var.set(False)
            self.use_edge_map_var.set(False)
            self.ppisp_freeze_var.set(False)
            self.debug_python_var.set(False)
            
            self.reset_options_to_defaults()
            
            self.log_message("All settings reset to hardcoded defaults.")
        
        self.folder_path.set("")
        self.output_text.delete(1.0, tk.END)
        self.log_text.delete(1.0, tk.END)
    
    def reset_options_to_defaults(self):
        """Reset all options to their default values"""
        default_values = [
            # Training options (0-11)
            ("", True), ("./output", True), ("30000", True), ("", False), ("images", False),
            ("", False), ("", False), ("", False), ("", False), ("", False), ("3840", False), ("auto", False),
            # Rendering options (12-18)
            ("3", False), ("", False), ("", False), ("", False), ("", False), ("50", False), ("", False),
            # Optimization options (19)
            ("mcmc", False),
            # Sparsity & Memory (20-23)
            ("1", False), ("15000", False), ("0.0005", False), ("0.6", False),
            # Logging & Masking (24-28)
            ("info", False), ("", False), ("", False), ("none", False), ("5678", False)
        ]
        
        for i, ((var, state_var), (val, enabled)) in enumerate(zip(zip(self.option_vars, self.option_states), default_values)):
            var.set(val)
            state_var.set(enabled)
    
    def generate_all_frames_batch(self):
        """Generate batch files for all frames in the specified range."""
        if not self.batch_mode_var.get():
            messagebox.showwarning("Warning", "Batch frame processing is only available in Batch Processing mode.")
            return
            
        try:
            start_num = int(self.start_frame_var.get())
            end_num = int(self.end_frame_var.get())
            
            if start_num > end_num:
                messagebox.showerror("Error", "Start frame number must be <= end frame number")
                return
                
            if start_num < 1 or end_num > 999:
                messagebox.showerror("Error", "Frame numbers must be between 001 and 999")
                return
                
        except ValueError:
            messagebox.showerror("Error", "Frame numbers must be valid integers")
            return
            
        try:
            nth = int(self.nth_frame_var.get())
            if nth < 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Every n-th must be a positive integer (>= 1)")
            return

        master_dir = self.master_folder_path.get()
        if not master_dir:
            messagebox.showerror("Error", "Please specify a master folder")
            return
            
        # Create BATCH folder if it doesn't exist
        batch_folder = os.path.join(master_dir, "BATCH")
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)
            self.log_message(f"Created BATCH folder: {batch_folder}")
        
        # Generate batch files for each frame
        generated_files = []
        base_output_name = self.base_output_name.get().strip() or ""
        
        self.log_message(f"Generating batch files for frames {start_num:03d} to {end_num:03d}...")
        self.log_message(f"Saving batch files to: {batch_folder}")
        
        for frame_num in range(start_num, end_num + 1):
            frame_prefix = "UFrame" if self.use_undistorted_var.get() else "Frame"
            frame_name = f"{frame_prefix}{frame_num:03d}"
            frame_folder = os.path.join(master_dir, frame_name)
            
            if not os.path.exists(frame_folder):
                self.log_message(f"Warning: Skipping {frame_name} - folder not found")
                continue
                
            original_folder_path = self.folder_path.get()
            self.folder_path.set(frame_folder)
            
            try:
                command_str = self.generate_command()
                output_paths = self.get_export_paths_to_create(use_full_paths=True)
                
                batch_content = "@echo off\n"
                batch_content += f"REM Generated LichtFeld Studio Batch File for {frame_name}\n"
                batch_content += "REM 3D Gaussian Splatting CUDA Implementation\n"
                batch_content += "REM\n\n"
                frame_folder_batch = frame_folder.replace('/', '\\')
                batch_content += f'cd /d "{frame_folder_batch}"\n'
                batch_content += "REM Create output folders if they don't exist\n"
                
                for output_path in output_paths:
                    batch_content += f'if not exist "{output_path}" mkdir "{output_path}"\n'
                
                batch_content += "\necho Starting LichtFeld Studio...\n"
                batch_content += command_str + "\n\n"
                batch_content += "if errorlevel 1 (\n"
                batch_content += f"    echo ERROR: Processing {frame_name} failed!\n"
                batch_content += "    echo Exiting with error code 1\n"
                batch_content += "    pause\n"
                batch_content += "    exit /b 1\n"
                batch_content += ")\n\n"
                batch_content += "echo.\n"
                batch_content += f"echo {frame_name} completed successfully!\n"
                batch_content += "echo.\n"
                
                batch_filename = os.path.join(batch_folder, f"{frame_name}_lichtfeld_batch.bat")
                with open(batch_filename, 'w') as f:
                    f.write(batch_content)
                    
                generated_files.append(batch_filename)
                self.log_message(f"Generated: {batch_filename}")
                
            except Exception as e:
                self.log_message(f"Error generating batch for {frame_name}: {str(e)}")
                
            finally:
                self.folder_path.set(original_folder_path)
        
        # Generate master batch file
        if generated_files:
            selected_frames = set(range(start_num, end_num + 1, nth))
            selected_files = []
            for batch_file in generated_files:
                m = re.search(r"(?:U)?Frame(\d{3})", os.path.basename(batch_file), re.IGNORECASE)
                if m and int(m.group(1)) in selected_frames:
                    selected_files.append(batch_file)

            if not selected_files:
                messagebox.showwarning("Warning", "No frames match the selected 'every n-th' setting.")
                return

            master_batch_lines = []
            master_batch_lines.append("@echo off")
            master_batch_lines.append("REM Master Batch File for LichtFeld Studio Frame Processing")
            master_batch_lines.append(f"REM Processing frames {start_num:03d} to {end_num:03d} (every {nth}th frame)")
            master_batch_lines.append("REM")
            master_batch_lines.append("")
            
            for i, batch_file in enumerate(selected_files, 1):
                frame_name = os.path.basename(batch_file).replace("_lichtfeld_batch.bat", "")
                master_batch_lines.append(f"echo.")
                master_batch_lines.append(f"echo ======================================")
                master_batch_lines.append(f"echo Processing {frame_name} ({i}/{len(selected_files)})")
                master_batch_lines.append(f"echo ======================================")
                master_batch_lines.append(f"echo.")
                master_batch_lines.append(f'call "{batch_file}"')
                master_batch_lines.append("")
                master_batch_lines.append(f"if errorlevel 1 (")
                master_batch_lines.append(f"    echo ERROR: Processing {frame_name} failed!")
                master_batch_lines.append(f"    echo Continuing with next frame...")
                master_batch_lines.append(f"    echo.")
                master_batch_lines.append(f")")
                master_batch_lines.append("")
            
            master_batch_lines.append(f"echo.")
            master_batch_lines.append(f"echo ======================================")
            master_batch_lines.append(f"echo All frame processing completed!")
            master_batch_lines.append(f"echo Processed {len(selected_files)} frames")
            master_batch_lines.append(f"echo ======================================")
            master_batch_lines.append(f"echo.")
            master_batch_lines.append(f"pause")
            
            master_batch_filename = os.path.join(batch_folder, f"master_lichtfeld_batch_Frame{start_num:03d}_to_Frame{end_num:03d}_{nth}.bat")
            master_batch_content = "\n".join(master_batch_lines)
            
            with open(master_batch_filename, 'w') as f:
                f.write(master_batch_content)
                
            self.log_message(f"Generated master batch file: {master_batch_filename}")
            
            if not self.expert_mode_var.get():
                success_msg = f"Successfully generated batch files for {len(generated_files)} frames!\n\n"
                success_msg += f"Individual batch files: {len(generated_files)}\n"
                success_msg += f"Master batch file (every {nth}th): {master_batch_filename}\n\n"
                success_msg += "Run the master batch file to process selected frames sequentially."
                
                messagebox.showinfo("Batch Generation Complete", success_msg)
        else:
            messagebox.showwarning("Warning", "No batch files were generated. Check that frame folders exist.")
    
    def save_config(self):
        """Save current settings to a JSON configuration file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save LichtFeld Studio Configuration",
            initialfile="lichtfeld_config.json",
            initialdir=script_dir
        )
        
        if file_path:
            if self.save_settings_to_json(file_path):
                if not self.expert_mode_var.get():
                    messagebox.showinfo("Success", f"Configuration saved successfully to:\n{file_path}")
    
    def load_config(self):
        """Load settings from a JSON configuration file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load LichtFeld Studio Configuration",
            initialdir=script_dir
        )
        
        if file_path:
            if self.load_settings_from_json(file_path):
                if not self.expert_mode_var.get():
                    messagebox.showinfo("Success", f"Configuration loaded successfully from:\n{file_path}")
    
    def get_all_option_names(self):
        """Get list of all option names in order"""
        return [
            # Training options (0-11)
            "--data-path", "--output-path", "--iter", "--max-cap", "--images",
            "--test-every", "--steps-scaler", "--init-num-pts", "--init-extent", "--init", "--max-width", "--resize_factor",
            # Rendering options (12-18)
            "--sh-degree", "--sh-degree-interval", "--min-opacity",
            "--import-cameras", "--timelapse-images", "--timelapse-every", "--ppisp-sidecar",
            # Optimization options (19)
            "--strategy",
            # Sparsity & Memory (20-23)
            "--tile-mode", "--sparsify-steps", "--init-rho", "--prune-ratio",
            # Logging & Masking (24-28)
            "--log-level", "--log-file", "--log-filter", "--mask-mode", "--debug-python-port"
        ]
    
    def load_settings_from_json(self, json_file_path=None):
        """Load settings from JSON file"""
        if json_file_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_file_path = os.path.join(script_dir, "lichtfeld_defaults.json")
        
        try:
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    settings = json.load(f)
                
                if "general" in settings:
                    general = settings["general"]
                    self.executable_path.set(general.get("executable_path", "LichtFeld-Studio.exe"))
                    self.master_folder_path.set(general.get("master_folder_path", "U:/CLI"))
                    self.base_output_name.set(general.get("base_output_name", ""))
                    self.use_frame_id_var.set(general.get("use_frame_id", True))
                    self.expert_mode_var.set(general.get("expert_mode", False))
                    self.project_path_var.set(general.get("config_path", ""))
                    self.view_ply_var.set(general.get("view_path", ""))
                    self.view_mode_var.set(general.get("view_mode", False))
                    self.resume_checkpoint_var.set(general.get("resume_checkpoint", ""))
                    
                    self.batch_mode_var.set(general.get("batch_mode", False))
                    self.start_frame_var.set(general.get("start_frame", "001"))
                    self.end_frame_var.set(general.get("end_frame", "010"))
                    self.nth_frame_var.set(general.get("nth_frame", "1"))
                    self.no_interop_var.set(general.get("no_interop", False))
                    self.eval_var.set(general.get("eval", False))
                    self.headless_var.set(general.get("headless", True))
                    self.train_var.set(general.get("train", False))
                    self.enable_mip_var.set(general.get("enable_mip", False))
                    self.bilateral_grid_var.set(general.get("bilateral_grid", False))
                    self.save_eval_images_var.set(general.get("save_eval_images", False))
                    self.save_depth_var.set(general.get("save_depth", False))
                    self.random_var.set(general.get("random", False))
                    self.gut_var.set(general.get("gut", False))
                    self.enable_sparsity_var.set(general.get("enable_sparsity", False))
                    self.bg_modulation_var.set(general.get("bg_modulation", False))
                    self.ppisp_var.set(general.get("ppisp", False))
                    self.ppisp_controller_var.set(general.get("ppisp_controller", False))
                    self.no_interop_var.set(general.get("no_interop", False))
                    self.invert_masks_var.set(general.get("invert_masks", False))
                    self.no_alpha_as_mask_var.set(general.get("no_alpha_as_mask", False))
                    self.no_cpu_cache_var.set(general.get("no_cpu_cache", False))
                    self.no_fs_cache_var.set(general.get("no_fs_cache", False))
                    self.use_error_map_var.set(general.get("use_error_map", False))
                    self.use_edge_map_var.set(general.get("use_edge_map", False))
                    self.ppisp_freeze_var.set(general.get("ppisp_freeze", False))
                    self.debug_python_var.set(general.get("debug_python", False))
                
                if "options" in settings:
                    options = settings["options"]
                    for i, (option_name, option_data) in enumerate(options.items()):
                        if i < len(self.option_vars):
                            self.option_vars[i].set(option_data.get("value", ""))
                            self.option_states[i].set(option_data.get("enabled", False))
                
                self.toggle_batch_mode()
                self.update_master_folder_paths()
                
                self.log_message(f"Settings loaded from: {json_file_path}")
                return True
            else:
                self.log_message(f"Settings file not found: {json_file_path}")
                return False
                
        except Exception as e:
            self.log_message(f"Error loading settings: {str(e)}")
            messagebox.showerror("Error", f"Failed to load settings:\n{str(e)}")
            return False
    
    def save_settings_to_json(self, json_file_path=None):
        """Save current settings to JSON file"""
        if json_file_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_file_path = os.path.join(script_dir, "lichtfeld_defaults.json")
        
        try:
            all_option_names = self.get_all_option_names()
            
            settings = {
                "general": {
                    "train": self.train_var.get(),
                    "no_interop": self.no_interop_var.get(),
                    "executable_path": self.executable_path.get(),
                    "master_folder_path": self.master_folder_path.get(),
                    "base_output_name": self.base_output_name.get(),
                    "use_frame_id": self.use_frame_id_var.get(),
                    "expert_mode": self.expert_mode_var.get(),
                    "config_path": self.project_path_var.get(),
                    "view_path": self.view_ply_var.get(),
                    "view_mode": self.view_mode_var.get(),
                    "resume_checkpoint": self.resume_checkpoint_var.get(),
                    "batch_mode": self.batch_mode_var.get(),
                    "start_frame": self.start_frame_var.get(),
                    "end_frame": self.end_frame_var.get(),
                    "nth_frame": self.nth_frame_var.get(),
                    "eval": self.eval_var.get(),
                    "headless": self.headless_var.get(),
                    "enable_mip": self.enable_mip_var.get(),
                    "bilateral_grid": self.bilateral_grid_var.get(),
                    "save_eval_images": self.save_eval_images_var.get(),
                    "save_depth": self.save_depth_var.get(),
                    "random": self.random_var.get(),
                    "gut": self.gut_var.get(),
                    "enable_sparsity": self.enable_sparsity_var.get(),
                    "bg_modulation": self.bg_modulation_var.get(),
                    "ppisp": self.ppisp_var.get(),
                    "ppisp_controller": self.ppisp_controller_var.get(),
                    "no_interop": self.no_interop_var.get(),
                    "invert_masks": self.invert_masks_var.get(),
                    "no_alpha_as_mask": self.no_alpha_as_mask_var.get(),
                    "no_cpu_cache": self.no_cpu_cache_var.get(),
                    "no_fs_cache": self.no_fs_cache_var.get(),
                    "use_error_map": self.use_error_map_var.get(),
                    "use_edge_map": self.use_edge_map_var.get(),
                    "ppisp_freeze": self.ppisp_freeze_var.get(),
                    "debug_python": self.debug_python_var.get()
                },
                "options": {}
            }
            
            for i, option_name in enumerate(all_option_names):
                if i < len(self.option_vars):
                    settings["options"][option_name] = {
                        "value": self.option_vars[i].get(),
                        "enabled": self.option_states[i].get()
                    }
            
            with open(json_file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.log_message(f"Settings saved to: {json_file_path}")
            return True
            
        except Exception as e:
            self.log_message(f"Error saving settings: {str(e)}")
            messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}")
            return False

def main():
    root = tk.Tk()
    app = LichtFeldStudioDialog(root)
    root.mainloop()

if __name__ == "__main__":
    main()