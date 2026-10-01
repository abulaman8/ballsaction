import os
import torch
from model_av_gate import AVGateNet
from train_av_gate import load_dataset_from_log, evaluate_gate_on_halves

def sweep_gate(checkpoint_path="checkpoints/av_gate_best.pth", test_csv="soccernet_eval_v3_log.csv"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = AVGateNet(in_features=15, hidden_dim=32, dropout=0.0).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded AVGate model from {checkpoint_path} (trained epoch {ckpt.get('epoch', 'N/A')})")
    print(f"Loading test data from {test_csv}...")
    test_halves = load_dataset_from_log(test_csv)
    print(f"Loaded {len(test_halves)} test halves.")
    
    print("\n" + "=" * 80)
    print("AV-GATE THRESHOLD SWEEP (EVALUATION ON 5 TEST MATCHES)")
    print("=" * 80)
    
    foul_thresholds = [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
    sp_thresholds = [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
    
    print("\n--- FOUL DETECTION PERFORMANCE ---")
    print(f"{'Thresh':<8} | {'TP':<4} {'FP':<4} {'FN':<4} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 60)
    
    best_foul_f1 = 0
    best_foul_row = None
    for th in foul_thresholds:
        res = evaluate_gate_on_halves(model, test_halves, foul_thresh=th, sp_thresh=0.85, device=device)
        f_m = res['foul']
        print(f"{th:<8.2f} | {f_m['tp']:<4} {f_m['fp']:<4} {f_m['fn']:<4} | {f_m['prec']*100:>8.2f}% | {f_m['rec']*100:>8.2f}% | {f_m['f1']*100:>8.2f}%")
        if f_m['f1'] > best_foul_f1:
            best_foul_f1 = f_m['f1']
            best_foul_row = (th, f_m)
            
    print("\n--- SET-PIECE DETECTION PERFORMANCE ---")
    print(f"{'Thresh':<8} | {'TP':<4} {'FP':<4} {'FN':<4} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 60)
    
    best_sp_f1 = 0
    best_sp_row = None
    for th in sp_thresholds:
        res = evaluate_gate_on_halves(model, test_halves, foul_thresh=0.85, sp_thresh=th, device=device)
        sp_m = res['sp']
        print(f"{th:<8.2f} | {sp_m['tp']:<4} {sp_m['fp']:<4} {sp_m['fn']:<4} | {sp_m['prec']*100:>8.2f}% | {sp_m['rec']*100:>8.2f}% | {sp_m['f1']*100:>8.2f}%")
        if sp_m['f1'] > best_sp_f1:
            best_sp_f1 = sp_m['f1']
            best_sp_row = (th, sp_m)
            
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS SUMMARY:")
    print("=" * 80)
    print(f"Best Foul: Thresh={best_foul_row[0]:.2f} | TP={best_foul_row[1]['tp']}, FP={best_foul_row[1]['fp']}, FN={best_foul_row[1]['fn']} | P={best_foul_row[1]['prec']*100:.2f}%, R={best_foul_row[1]['rec']*100:.2f}%, F1={best_foul_row[1]['f1']*100:.2f}%")
    print(f"Best SetPiece: Thresh={best_sp_row[0]:.2f} | TP={best_sp_row[1]['tp']}, FP={best_sp_row[1]['fp']}, FN={best_sp_row[1]['fn']} | P={best_sp_row[1]['prec']*100:.2f}%, R={best_sp_row[1]['rec']*100:.2f}%, F1={best_sp_row[1]['f1']*100:.2f}%")

if __name__ == "__main__":
    sweep_gate()
