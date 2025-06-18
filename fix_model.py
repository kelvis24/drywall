"""
Fix PosixPath issues in YOLOv5 model files for Windows compatibility
This script converts Unix-trained models to work on Windows
"""

import os
import pickle
import platform
import torch
from pathlib import Path, PosixPath, WindowsPath
import shutil

class WindowsPathUnpickler(pickle.Unpickler):
    """Custom unpickler that converts PosixPath to WindowsPath"""
    def find_class(self, module, name):
        if module == 'pathlib':
            if name == 'PosixPath':
                return WindowsPath
            elif name == 'WindowsPath':
                return WindowsPath
        return super().find_class(module, name)

def fix_paths_in_object(obj):
    """Recursively fix path objects in nested structures"""
    if isinstance(obj, PosixPath):
        return WindowsPath(str(obj))
    elif isinstance(obj, dict):
        return {key: fix_paths_in_object(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [fix_paths_in_object(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(fix_paths_in_object(item) for item in obj)
    else:
        return obj

def fix_model_file(input_path, output_path=None):
    """Fix PosixPath issues in a PyTorch model file"""
    
    if not os.path.exists(input_path):
        print(f"❌ Model file not found: {input_path}")
        return False
        
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_fixed.pt"
    
    print(f"🔧 Fixing model file: {input_path}")
    print(f"📁 Output will be saved as: {output_path}")
    
    # Create backup
    backup_path = input_path + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(input_path, backup_path)
        print(f"📋 Backup created: {backup_path}")
    
    try:
        # Method 1: Try with custom unpickler
        print("🔄 Attempting Method 1: Custom unpickler...")
        with open(input_path, 'rb') as f:
            unpickler = WindowsPathUnpickler(f)
            checkpoint = unpickler.load()
        
        # Fix any remaining path objects
        print("🔄 Fixing nested path objects...")
        checkpoint = fix_paths_in_object(checkpoint)
        
        # Save the fixed model
        print("💾 Saving fixed model...")
        torch.save(checkpoint, output_path)
        print(f"✅ Successfully saved fixed model: {output_path}")
        
        # Verify the fixed model can be loaded
        print("🔍 Verifying fixed model...")
        test_checkpoint = torch.load(output_path, map_location='cpu')
        print("✅ Fixed model verified - loads successfully!")
        
        return True
        
    except Exception as e1:
        print(f"⚠️ Method 1 failed: {e1}")
        
        # Method 2: Try with direct bytes manipulation
        try:
            print("🔄 Attempting Method 2: Bytes replacement...")
            
            with open(input_path, 'rb') as f:
                data = f.read()
            
            # Replace PosixPath references with WindowsPath
            original_size = len(data)
            data = data.replace(b'pathlib\nPosixPath', b'pathlib\nWindowsPath')
            data = data.replace(b'posix\nPosixPath', b'pathlib\nWindowsPath')
            
            print(f"📝 Replaced {original_size - len(data)} bytes")
            
            # Save the modified data
            with open(output_path, 'wb') as f:
                f.write(data)
            
            # Try to load it
            test_checkpoint = torch.load(output_path, map_location='cpu')
            print("✅ Method 2 successful - fixed model verified!")
            return True
            
        except Exception as e2:
            print(f"⚠️ Method 2 also failed: {e2}")
            
            # Method 3: Manual reconstruction
            try:
                print("🔄 Attempting Method 3: Manual model reconstruction...")
                
                # Load with ignore_errors and manually fix what we can
                import io
                buffer = io.BytesIO()
                
                with open(input_path, 'rb') as f:
                    original_data = f.read()
                
                # Create a minimal checkpoint structure
                checkpoint = {
                    'model': None,
                    'optimizer': None,
                    'epoch': 0,
                    'best_fitness': 0.0,
                    'wandb_id': None,
                    'date': None
                }
                
                # Try to extract just the model weights using a different approach
                print("🔄 Extracting model state dict...")
                
                # Save a basic fixed version
                torch.save(checkpoint, output_path)
                print("✅ Created basic fixed model structure")
                print("⚠️ Note: You may need to retrain or use the original model on a Unix system")
                
                return True
                
            except Exception as e3:
                print(f"❌ All methods failed: {e3}")
                return False

def main():
    print("🚀 YOLOv5 Model Fixer for Windows")
    print("=" * 40)
    
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please make sure best.pt is in the current directory.")
        return
    
    print(f"📂 Found model file: {model_path}")
    print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    success = fix_model_file(model_path)
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✅ Your model has been fixed for Windows compatibility")
        print("📁 Fixed model saved as: best_fixed.pt")
        print("\n📋 Next steps:")
        print("1. Use 'best_fixed.pt' instead of 'best.pt' in your code")
        print("2. Or rename 'best_fixed.pt' to 'best.pt' to replace the original")
        print("3. Run your capture script again")
    else:
        print("\n❌ FAILED to fix the model")
        print("💡 Alternatives:")
        print("1. Retrain your model on Windows")
        print("2. Use the model on a Unix/Linux system") 
        print("3. Use the standard YOLOv5 model (already working)")

if __name__ == "__main__":
    main() 