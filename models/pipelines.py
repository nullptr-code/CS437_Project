import torch
import copy
from tqdm.notebook import tqdm
import os
import cv2

from util import threshold



def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    val_loss_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    last_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch} / {num_epochs - 1}", end="\t")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            elif phase == "val":
                model.eval()

            running_loss = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} loss: {epoch_loss:.4f}", end="\t")

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join("weights", "best.pt"))

            if phase == "val":
                val_loss_history.append(epoch_loss)
                last_model_wts = copy.deepcopy(model.state_dict())
                torch.save(last_model_wts, os.path.join("weights", "last.pt"))

            if phase == "train":
                train_loss_history.append(epoch_loss)

        print()

    print("-" * 30)
    print(f"Training Complete")
    print(f"Best Validation Loss: {best_loss:.4f}")

    return val_loss_history, train_loss_history

def test_model(model, test_paths, label_paths, data_transforms,device, ROWS=256, COLS=256):
    outputs = []
    inputs = []
    labels = []

    for image_path, label_path in zip(test_paths, label_paths):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        print(image.shape)
        break
        inputs.append(cv2.resize(image, (256, 256)))
        image = data_transforms["image"](image).to(device=device)
        image = image.reshape(1, *image.shape)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        labels.append(label)
        output = model(image)[0].permute(1, 2, 0).detach().cpu().numpy()
        
        outputs.append(threshold(output))
    return outputs, inputs, labels