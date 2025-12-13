import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix


class BookModelVisualizer:
    def __init__(self, model, device, class_names):
        """
        Utility per visualizzare predizioni, Grad-CAM e metriche
        del modello su copertine di libri.

        Args:
            model: modello PyTorch già caricato (stessa architettura usata in training).
            device: torch.device('cuda') o torch.device('cpu').
            class_names: lista di stringhe con i nomi delle classi
                         nello stesso ordine usato da train_val_dataset.classes.
        """
        self.model = model
        self.device = device
        self.class_names = class_names

        # Mean / std usate anche nel notebook per Normalize
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    # ------------------------------------------------------------------
    #  Denormalizzazione immagine
    # ------------------------------------------------------------------
    def denormalize(self, tensor):
        """
        Converte un tensore normalizzato (C, H, W) in immagine NumPy (H, W, C)
        con valori in [0,1], pronta per plt.imshow.
        """
        img = tensor.cpu().numpy()          # (C, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        img = img * self.std + self.mean    # denormalizzazione
        img = np.clip(img, 0, 1)
        return img

    # ------------------------------------------------------------------
    #  Visualizzazione predizioni Top-3 su immagini random
    # ------------------------------------------------------------------
    def plot_predictions(self, dataset, num_samples=5):
        """
        Seleziona num_samples immagini a caso dal dataset,
        mostra le copertine, la label reale e le Top-3 predizioni del modello.

        dataset: tipicamente val_dataset o test_dataset
        num_samples: quante immagini mostrare
        """
        self.model.eval()

        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        num_cols = len(indices)
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

        if num_cols == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            image, label_idx = dataset[idx]  # image: tensor (C,H,W), label_idx: int
            ax = axes[i]

            input_tensor = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)        # logits [1, num_classes]
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, 3, dim=1)

            preds = top_indices[0].cpu().numpy()
            probs_np = top_probs[0].cpu().numpy()

            img_np = self.denormalize(image)
            ax.imshow(img_np)
            ax.axis("off")

            true_label_name = self.class_names[label_idx]
            top1_idx = int(preds[0])
            top1_name = self.class_names[top1_idx]

            if top1_idx == label_idx:
                status = "TOP-1 ✅"
                status_color = "green"
            elif int(label_idx) in preds:
                status = "TOP-3 ⚠️"
                status_color = "orange"
            else:
                status = "FAIL ❌"
                status_color = "red"

            title = f"True: {true_label_name}\nPred: {top1_name}\n{status}"
            ax.set_title(title, color=status_color, fontsize=12, fontweight="bold")

            info_text = ""
            for j in range(3):
                cls_name = self.class_names[int(preds[j])]
                prob_val = probs_np[j] * 100
                marker = "👈" if int(preds[j]) == int(label_idx) else ""
                info_text += f"{j+1}. {cls_name}: {prob_val:.1f}% {marker}\n"

            ax.text(
                0.5, -0.1, info_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
            )

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Singolo esempio (senza Grad-CAM)
    # ------------------------------------------------------------------
    def visualize_single_prediction(self, image_tensor, true_label_idx=None):
        """
        Visualizza una singola immagine (normalizzata) con le Top-3 predizioni.

        image_tensor: tensore (C,H,W) già normalizzato (stesse trasformazioni del dataset).
        true_label_idx: indice della label reale (opzionale).
        """
        self.model.eval()

        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, 3, dim=1)

        preds = top_indices[0].cpu().numpy()
        probs_np = top_probs[0].cpu().numpy()

        img_np = self.denormalize(image_tensor)

        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.axis("off")

        # Titolo: true vs pred top-1
        if true_label_idx is not None:
            true_name = self.class_names[true_label_idx]
        else:
            true_name = "Unknown"

        top1_idx = int(preds[0])
        top1_name = self.class_names[top1_idx]

        if true_label_idx is None:
            status = "Pred only"
            status_color = "black"
        elif top1_idx == true_label_idx:
            status = "TOP-1 ✅"
            status_color = "green"
        elif int(true_label_idx) in preds:
            status = "TOP-3 ⚠️"
            status_color = "orange"
        else:
            status = "FAIL ❌"
            status_color = "red"

        title = f"True: {true_name}\nPred: {top1_name}\n{status}"
        plt.title(title, color=status_color, fontsize=12, fontweight="bold")

        # Top-3 sotto forma di textbox
        info_text = ""
        for j in range(3):
            cls_name = self.class_names[int(preds[j])]
            prob_val = probs_np[j] * 100
            marker = ""
            if true_label_idx is not None and int(preds[j]) == int(true_label_idx):
                marker = "👈"
            info_text += f"{j+1}. {cls_name}: {prob_val:.1f}% {marker}\n"

        plt.gcf().text(
            0.5, 0.02, info_text,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Grad-CAM: generazione della heatmap per una singola immagine
    # ------------------------------------------------------------------
    def generate_gradcam(self, input_tensor, target_class=None, target_layer_name="layer4"):
        """
        Calcola Grad-CAM per una singola immagine (C,H,W) già normalizzata (ImageNet).

        Args:
            input_tensor: tensore immagine (C,H,W) normalizzato.
            target_class: indice della classe bersaglio; se None usa la classe predetta.
            target_layer_name: nome del layer convoluzionale su cui calcolare Grad-CAM
                               (per ResNet50 tipicamente 'layer4').

        Ritorna:
            cam: array 2D numpy in [0,1] con shape [H, W]
            target_class: indice intero della classe bersaglio usata per il backward.
        """
        self.model.eval()

        # [1, C, H, W]
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        activations = {}
        gradients = {}

        def fwd_hook(module, inp, out):
            activations["value"] = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        target_layer = dict(self.model.named_modules())[target_layer_name]
        handle_fwd = target_layer.register_forward_hook(fwd_hook)

        if hasattr(target_layer, "register_full_backward_hook"):
            handle_bwd = target_layer.register_full_backward_hook(bwd_hook)
        else:
            handle_bwd = target_layer.register_backward_hook(bwd_hook)

        output = self.model(input_tensor)  # [1, num_classes]

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        handle_fwd.remove()
        handle_bwd.remove()

        acts = activations["value"]   # [1, C, H', W']
        grads = gradients["value"]    # [1, C, H', W']

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()  # [H, W]

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, target_class

    # ------------------------------------------------------------------
    #  Grad-CAM: visualizzazione immagine + heatmap
    # ------------------------------------------------------------------
    def show_gradcam_on_image(self, input_tensor, cam,
                              pred_class_idx=None, true_label_idx=None, example_idx=None):
        """
        input_tensor: tensore immagine singola (C,H,W), normalizzato.
        cam: array numpy 2D [H,W] in [0,1].
        pred_class_idx: indice della classe predetta (per il titolo del pannello destro).
        true_label_idx: indice della label reale (per il pannello sinistro).
        example_idx: indice progressivo per etichetta nel titolo.
        """
        img = self.denormalize(input_tensor)  # (H,W,C)

        plt.figure(figsize=(6, 3))

        # Immagine originale + true label
        plt.subplot(1, 2, 1)
        title_left = "Original"
        if example_idx is not None:
            title_left += f" #{example_idx}"
        if true_label_idx is not None:
            try:
                title_left += f"\nTrue: {self.class_names[true_label_idx]}"
            except Exception:
                title_left += f"\nTrue idx: {true_label_idx}"
        plt.title(title_left)
        plt.imshow(img)
        plt.axis("off")

        # Immagine + heatmap
        plt.subplot(1, 2, 2)
        title = "Grad-CAM"
        if pred_class_idx is not None:
            try:
                title += f"\nPred: {self.class_names[pred_class_idx]}"
            except Exception:
                title += f"\nPred idx: {pred_class_idx}"
        plt.title(title)
        plt.imshow(img)
        plt.imshow(cam, cmap="jet", alpha=0.4)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Grad-CAM: esempi multi-immagine su dataset (val/test)
    # ------------------------------------------------------------------
    def plot_gradcam_examples(self, dataset, num_examples=5, seed=None, target_layer_name="layer4"):
        """
        Seleziona alcune immagini random dal dataset (val/test),
        calcola Grad-CAM e mostra immagine + heatmap per ognuna.

        Args:
            dataset: tipicamente val_dataset o test_dataset (stesse trasformazioni del training).
            num_examples: quante immagini mostrare.
            seed: se specificato (int), rende la scelta degli esempi riproducibile.
            target_layer_name: layer su cui calcolare Grad-CAM (default: 'layer4').
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        num_to_show = min(num_examples, len(dataset))
        indices = random.sample(range(len(dataset)), num_to_show)

        for example_idx, idx in enumerate(indices, start=1):
            sample_img, sample_label = dataset[idx]  # immagine già trasformata (resize + normalize)
            cam, pred_cls = self.generate_gradcam(sample_img, target_layer_name=target_layer_name)

            print(f"\n=== Example {example_idx} (dataset index {idx}) ===")
            try:
                true_name = self.class_names[sample_label]
            except Exception:
                true_name = f"idx {sample_label}"
            try:
                pred_name = self.class_names[pred_cls]
            except Exception:
                pred_name = f"idx {pred_cls}"

            print(f"True label:      {sample_label} ({true_name})")
            print(f"Predicted class: {pred_cls} ({pred_name})")

            self.show_gradcam_on_image(
                sample_img,
                cam,
                pred_class_idx=pred_cls,
                true_label_idx=sample_label,
                example_idx=example_idx
            )

    # ------------------------------------------------------------------
    #  Confusion Matrix (val o test)
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, dataloader, normalize=False,
                              title="Matrice di Confusione - Generi Letterari"):
        """
        Calcola e visualizza la matrice di confusione su un dataloader
        (tipicamente validation o test).

        Args:
            dataloader: DataLoader (es. test_loader).
            normalize: se True, normalizza per riga (percentuali per classe).
            title: titolo del grafico.
        """
        self.model.eval()
        y_true = []
        y_pred = []

        print("📊 Calcolo delle predizioni per la Matrice di Confusione...")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        plt.figure(figsize=(20, 16))  # Dimensioni grandi per 30 classi
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )

        plt.ylabel('Vero Genere (True Label)', fontsize=14)
        plt.xlabel('Genere Predetto (Predicted Label)', fontsize=14)
        plt.title(title, fontsize=18)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return cm
