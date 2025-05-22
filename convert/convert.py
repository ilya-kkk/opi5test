import torch

class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

# Подменяем все возможные модули на заглушки
import ultralytics.nn.modules.block as block
import ultralytics.nn.modules.head as head
import ultralytics.nn.modules.conv as conv

# Заглушка по умолчанию
def dummy_setattr(module, name):
    if not hasattr(module, name):
        setattr(module, name, DummyModule)

# Подставим заглушки "во всё что можно"
block_names = ['SCDown', 'PSA', 'Attention', 'C2fCIB', 'CIB', 'RepVGGDW']
head_names = ['v10Detect']
for name in block_names:
    dummy_setattr(block, name)
for name in head_names:
    dummy_setattr(head, name)

# Теперь пробуем загрузить
model_path = "v10nfull.pt"
ckpt = torch.load(model_path, map_location="cpu")
print("Checkpoint loaded!")
