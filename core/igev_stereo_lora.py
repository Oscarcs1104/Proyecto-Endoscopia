from core import igev_stereo
from igev_stereo import IGEVStereo
from core.utils.args import Args
from Dataset.dataloader import get_scared_dataloader

args = Args()

model = igev_stereo.IGEVStereo(args) 

target_modules_candidates = [
    ".conv",  # Catches many convolutional layers
    ".conv1",
    ".conv2",
    ".conv3",
    ".q_proj", # If there are query projections in attention
    ".v_proj", # If there are value projections in attention
    ".k_proj", # If there are key projections in attention
    "g_proj", # Example, if some specific projection exists
    "c_proj",
    "attn_q", # If they have custom names like this
    "attn_v",
    "attn_k",
    "gru.weight_ih", # Convolutional GRU weights
    "gru.weight_hh"
]

actual_target_modules = []
for name, module in model.named_modules():
    for candidate in target_modules_candidates:
        if candidate in name and isinstance(module, (nn.Linear, nn.Conv2d)):
            # Filter by actual layer types LoRA can modify
            actual_target_modules.append(name)
            # Break to avoid adding the same module multiple times if multiple candidates match
            break
print(f"Discovered LoRA target modules: {actual_target_modules}")
print(f"Total LoRA target modules: {len(actual_target_modules)}")

from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
    target_modules=actual_target_modules,
    lora_dropout=0.1,
    bias="none",
    r=4,
    lora_alpha=16,
    task_type=TaskType.FEATURE_EXTRACTION,

)

model_with_lora = get_peft_model(model, peft_config)
model_with_lora.print_trainable_parameters()

train_loader, val_loader, total_size = get_scared_dataloader(args, train=True)

