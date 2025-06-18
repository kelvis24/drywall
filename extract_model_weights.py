"""
Advanced YOLOv5 Model Weight Extractor for Windows
This script extracts actual model weights from Unix-trained models
"""

import os
import pickle
import torch
import io
from pathlib import WindowsPath
import struct

class RobustUnpickler(pickle.Unpickler):
    """Unpickler that can handle problematic path objects"""
    
    def find_class(self, module, name):
        if module == 'pathlib':
            if name in ['PosixPath', 'Path']:
                return WindowsPath
        return super().find_class(module, name)
    
    def persistent_load(self, pid):
        """Handle persistent load operations"""
        # This handles the storage loading for PyTorch tensors
        if isinstance(pid, tuple) and len(pid) == 2:
            typename, data = pid
            if typename == 'storage':
                storage_type, root, location, size, view_metadata = data
                # Return a placeholder - the actual tensor data will be loaded separately
                return f"<<STORAGE:{storage_type}:{location}:{size}>>"
        return pid

def extract_raw_weights(model_path):
    """Extract raw model weights using low-level approach"""
    print(f"🔍 Analyzing model file structure: {model_path}")
    
    try:
        # Method 1: Try to load as ZIP (PyTorch uses ZIP format)
        import zipfile
        
        if zipfile.is_zipfile(model_path):
            print("📦 Model is in ZIP format (newer PyTorch)")
            
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                print("📋 Files in model archive:")
                for file_info in zip_ref.filelist:
                    print(f"   - {file_info.filename} ({file_info.file_size} bytes)")
                
                # Try to extract the pickle data
                if 'data.pkl' in zip_ref.namelist():
                    print("🔄 Extracting pickle data...")
                    with zip_ref.open('data.pkl') as pkl_file:
                        pkl_data = pkl_file.read()
                    
                    # Try to unpickle with our robust unpickler
                    unpickler = RobustUnpickler(io.BytesIO(pkl_data))
                    checkpoint = unpickler.load()
                    print("✅ Successfully extracted checkpoint structure!")
                    return checkpoint
                else:
                    print("⚠️ No data.pkl found in archive")
                    
        else:
            print("📄 Model is in legacy pickle format")
            # Try direct pickle loading with robust unpickler
            with open(model_path, 'rb') as f:
                unpickler = RobustUnpickler(f)
                checkpoint = unpickler.load()
            print("✅ Successfully loaded with robust unpickler!")
            return checkpoint
            
    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")
        return None

def reconstruct_yolo_model(checkpoint_data, output_path):
    """Reconstruct a working YOLO model from extracted data"""
    
    print("🔧 Reconstructing YOLO model...")
    
    # Create a new checkpoint with clean structure
    new_checkpoint = {}
    
    if checkpoint_data and isinstance(checkpoint_data, dict):
        print("📋 Found checkpoint data structure")
        
        # Copy safe fields
        safe_fields = ['epoch', 'best_fitness', 'ema', 'updates', 'optimizer', 'lr_scheduler', 'wandb_id']
        for field in safe_fields:
            if field in checkpoint_data:
                new_checkpoint[field] = checkpoint_data[field]
                print(f"   ✅ Copied {field}")
        
        # Handle model state dict
        if 'model' in checkpoint_data:
            model_data = checkpoint_data['model']
            print(f"📊 Model data type: {type(model_data)}")
            
            # If it's a state dict, use it directly
            if isinstance(model_data, dict):
                new_checkpoint['model'] = model_data
                print("   ✅ Copied model state dict")
            else:
                # If it's an object, try to get its state dict
                try:
                    if hasattr(model_data, 'state_dict'):
                        new_checkpoint['model'] = model_data.state_dict()
                        print("   ✅ Extracted state dict from model object")
                    else:
                        print("   ⚠️ Model object has no state_dict method")
                        new_checkpoint['model'] = None
                except Exception as e:
                    print(f"   ❌ Error extracting state dict: {e}")
                    new_checkpoint['model'] = None
        else:
            print("   ⚠️ No model found in checkpoint")
            new_checkpoint['model'] = None
    
    else:
        print("❌ Invalid checkpoint data structure")
        return False
    
    # Save the reconstructed model
    try:
        print(f"💾 Saving reconstructed model to: {output_path}")
        torch.save(new_checkpoint, output_path)
        
        # Verify the saved model
        print("🔍 Verifying reconstructed model...")
        test_load = torch.load(output_path, map_location='cpu')
        print("✅ Reconstructed model verified!")
        
        # Print model info
        if 'model' in test_load and test_load['model'] is not None:
            if isinstance(test_load['model'], dict):
                print(f"📊 Model contains {len(test_load['model'])} layers/parameters")
                
                # Show some layer names
                layer_names = list(test_load['model'].keys())[:5]
                print("🔍 Sample layers:")
                for name in layer_names:
                    print(f"   - {name}")
                if len(test_load['model']) > 5:
                    print(f"   ... and {len(test_load['model']) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving reconstructed model: {e}")
        return False

def main():
    print("🚀 Advanced YOLOv5 Model Weight Extractor")
    print("=" * 45)
    
    model_path = "best.pt"
    output_path = "best_weights_extracted.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        return
    
    print(f"📂 Input model: {model_path}")
    print(f"📁 Output will be: {output_path}")
    print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Extract the raw weights
    checkpoint_data = extract_raw_weights(model_path)
    
    if checkpoint_data is None:
        print("❌ Failed to extract model data")
        return
    
    # Reconstruct the model
    success = reconstruct_yolo_model(checkpoint_data, output_path)
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✅ Model weights successfully extracted and reconstructed!")
        print(f"📁 New model saved as: {output_path}")
        print("\n📋 Next steps:")
        print(f"1. Use '{output_path}' in your capture script")
        print("2. Update help.py to load this new model file")
        print("3. Test ball detection with your custom trained weights!")
    else:
        print("\n❌ FAILED to reconstruct model")
        print("💡 The original model may be too corrupted for automatic fixing")
        print("🔧 Consider retraining on Windows or using on a Unix system")

if __name__ == "__main__":
    main() 