import torch
import torch.nn as nn
from model import X3DFreeKickModel
import gc

def find_max_batch_size():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available. Cannot test GPU VRAM.")
        return 2

    # Instantiate model
    try:
        model = X3DFreeKickModel(num_classes=2, pretrained=False).to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return 2
    
    batch_size = 1
    max_successful_batch = 1
    
    print("Testing maximum batch size...")
    while True:
        try:
            # Create dummy data (B, C, T, H, W)
            dummy_videos = torch.randn(batch_size, 3, 40, 224, 224, device=device)
            dummy_targets = torch.randint(0, 2, (batch_size,), device=device)
            
            optimizer.zero_grad()
            outputs = model(dummy_videos)
            loss = criterion(outputs, dummy_targets)
            loss.backward()
            optimizer.step()
            
            print(f"Batch size {batch_size} succeeded.")
            max_successful_batch = batch_size
            batch_size += 1
            
            # Clean up memory for the next iteration
            del dummy_videos, dummy_targets, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM on batch size {batch_size}.")
                break
            else:
                print(f"Unexpected runtime error: {e}")
                break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
            
    print(f"\nMaximum stable batch size determined: {max_successful_batch}")
    # We should probably return max_successful_batch - 1 or so to be safe during real training, 
    # but the user asked for what we can handle right now.
    # To be perfectly safe across epochs with real data loading, we use exactly max_successful_batch
    # or max_successful_batch - 1 if it's large.
    return max_successful_batch

if __name__ == "__main__":
    max_bs = find_max_batch_size()
    
    # Let's save it to a file so the bash command can easily read it
    with open("max_bs.txt", "w") as f:
        f.write(str(max_bs))
