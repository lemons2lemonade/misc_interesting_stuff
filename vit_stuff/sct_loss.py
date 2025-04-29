import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from tcg_dataloader import TripletDataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm

# Vision Transformer (ViT) Model without classification head
class SimpleViT(nn.Module):
    def __init__(self):
        super(SimpleViT, self).__init__()
        self.vit = AutoModelForImageClassification.from_pretrained("google/vit-large-patch16-384")
        self.vit.head = nn.Identity()  # remove classification head

    def forward(self, x):
        return self.vit(x)  # return embeddings from ViT

class SCTLoss(Module):
    """
    Compute the loss for selective contrastive triplet (SCT) learning 
    along with hard negative mining options.

    Attributes:
    - sct (bool): If True, use SCT setting. Otherwise, use standard triplet loss.
    - semi (bool): If True, mask out non-semi-hard negatives.
    - lam (float): Weight for hard triplet loss when using SCT setting.
    """

    def __init__(self):
        self.margin = 0.2


    def _compute_cosine_similarity(self, fvec):
        """
        Compute the cosine similarity for feature vectors.

        Args:
        - fvec (torch.Tensor): A 2D tensor of shape (batch_size, feature_size) with feature vectors.

        Returns:
        - torch.Tensor: A 2D tensor of shape (batch_size, batch_size) with pairwise cosine similarities.
        """
        # Normalize the feature vectors to unit length
        fvec_norm = fvec / (fvec.norm(dim=1, keepdim=True) + 1e-8)  # Added 1e-8 for numerical stability

        # Compute cosine similarity
        return fvec_norm @ fvec_norm.t()


    def sct_loss_with_semi_hard_negatives(self, anchor_embedding_batch, positive_embedding_batch, negative_embedding_batch):
        """
        Custom loss function with hard and semi-hard negatives.

        Args:
        - anchor_embedding (torch.Tensor): Embeddings for the anchor images.
        - positive_embedding (torch.Tensor): Embeddings for the positive images.
        - negative_embedding (torch.Tensor): Embeddings for the negative images.
        - margin (float): Margin for distinguishing hard negatives.

        Returns:
        - torch.Tensor: The computed loss with considerations for hard and semi-hard negatives.
        """
        # Compute cosine similarities
        positive_similarity = F.cosine_similarity(anchor_embedding_batch.logits, positive_embedding_batch.logits, dim=1)
        negative_similarity = F.cosine_similarity(anchor_embedding_batch.logits, negative_embedding_batch.logits, dim=1)

        # Compute the base triplet loss
        triplet_loss = F.relu(self.margin - positive_similarity + negative_similarity)

        # Identify hard negatives (negative similarity greater than positive similarity minus margin)
        hard_negatives = negative_similarity > (positive_similarity - self.margin)

        # Identify semi-hard negatives (negative similarity within semi_hard_margin of the positive similarity)
        semi_hard_negatives = (negative_similarity > (positive_similarity - self.margin)) & (negative_similarity <= positive_similarity)
        
        # Convert the Boolean tensor 'hard_negatives' to a floating-point tensor before computing the mean
        hard_negatives_float = hard_negatives.float()
        # Now you can compute the mean without encountering the dtype error
        hard_negatives_mean = hard_negatives_float.mean()

        # Or, the functions can be chained for a 1-liner
        semi_hard_negatives_mean = semi_hard_negatives.float().mean()

        print("Semi-Hard Negatives Loss: ", semi_hard_negatives_mean)
        print("Hard Negatives Loss: ", hard_negatives_mean)
        print("Triplet Loss: ", triplet_loss.mean())
        # Apply additional weight to hard and semi-hard negatives
        weighted_loss = triplet_loss * (1 + hard_negatives.float() + semi_hard_negatives.float())

        # Calculate mean loss
        loss = weighted_loss.mean()

        return loss
    

def main(): 
    # Model and Optimizer
    model = SimpleViT()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Each item is a tuple of batch_size batches
    # labels are used for generating similarity scores required for SCTLoss
    jsonl_file = 'labeled_bulbasaur_triplets.jsonl'
    base_path = 'cropped_images/'
    triplet_dataset = TripletDataset(jsonl_file, base_path)

    batch_num = 4
    shuffle = True
    num_workers = 12
    data_loader = DataLoader(triplet_dataset, batch_num, shuffle, num_workers=num_workers, pin_memory=True)
    num_epochs = 100

    sct_loss = SCTLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Training Loop
    for epoch in tqdm(range(num_epochs)):
        counter = 0  # Initialize counter for batches
        for element in tqdm(data_loader):
            element = [t.to(device) if torch.is_tensor(t) else t for t in element] #  put batch tensors (only) on the GPU

            optimizer.zero_grad()

            anchor = element[1]
            positive = element[2]
            negative = element[3]
        
            print(" \n PASSING TENSORS TO MODEL...")
            # Forward Pass - Obtain embeddings from ViT for each element of the triplet
            anchor_embeddings = model(anchor)
            positive_embeddings = model(positive)
            negative_embeddings = model(negative)

            # put batch onto GPU
            anchor_embeddings.logits.to(device)
            positive_embeddings.logits.to(device)
            negative_embeddings.logits.to(device)
        
            print("\n CALCULATING LOSS FUNCTION...")
            # Calculate Loss using SCTLoss
            loss = sct_loss.sct_loss_with_semi_hard_negatives(anchor_embeddings, positive_embeddings, negative_embeddings)
        
            # Backward Pass and Optimization
            loss.backward()
            optimizer.step()
        
            # Increment the counter
            counter += 1

            # Print/Log loss if needed
            print_interval = 10  # You can adjust this interval to your preference
            if counter % print_interval == 0:
                print(f"Epoch: {epoch+1}, Batch: {counter}/{len(data_loader)}, Loss: {loss.item()}")
    

    # Save the model and optimizer state at the end of training
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }

    # Define the path where the checkpoint is saved
    checkpoint_path = "first_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()