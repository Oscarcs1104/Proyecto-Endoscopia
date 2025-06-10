from core import igev_stereo
from core.igev_stereo import IGEVStereo
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn

target_modules_candidates = [
    ".conv",  # Catches many convolutional layers
    ".conv1",
    ".conv2",
    ".conv3",
    "c_proj",
    "attn_q", # If they have custom names like this
    "attn_v",
    "attn_k",
    "gru.weight_ih", # Convolutional GRU weights
    "gru.weight_hh"
    ".gru"
]

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict

def IGEVStereoLoraModel(arguments):

    actual_target_modules = []
    model = igev_stereo.IGEVStereo(arguments) # Instantiate the model
    checkpoint = torch.load('./Checkpoints/sceneflow.pth', map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    for name, module in model.named_modules():
        for candidate in target_modules_candidates:
            if candidate in name and isinstance(module, (nn.Linear, nn.Conv2d)):
                # Filter by actual layer types LoRA can modify
                actual_target_modules.append(name)
                # Break to avoid adding the same module multiple times if multiple candidates match
                break
    #print(f"Discovered LoRA target modules: {actual_target_modules}")
    #print(f"Total LoRA target modules: {len(actual_target_modules)}")

    peft_config = LoraConfig(
        target_modules=actual_target_modules,
        lora_dropout=0.1,
        bias="none",
        r=4,
        lora_alpha=16,
        task_type=None
    )

    model_with_lora = get_peft_model(model, peft_config)

    #for name, param in model_with_lora.named_parameters():
    #    param.requires_grad = "lora_" in name
    model_with_lora.print_trainable_parameters()
    return model_with_lora
