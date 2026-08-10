# **Urban Scene Conditioning for Stable Diffusion 3.5**

This academic project aimed to explore a solution for generating more realistic and structurally consistent urban scenes with Stable Diffusion 3.5, for example by reducing unrealistic object placements such as a car appearing in the sky.

## **Overview**

### **Embeddings**

* The semantic segmentation map provides a visual representation of the image. Each object category is associated with a specific color, which helps preserve information about object position, shape, size, and spatial organization.
* The GNN provides a vector representation of the elements contained in the image. It represents the scene as nodes and directed relationships between them. In the current implementation, the relation labels themselves are not used; only the direction of the connections is preserved.
* To combine both representations, the segmentation map is first encoded into an embedding. The segmentation embeddings, GNN embeddings, and fused embeddings are then stored separately in three different files.

### **Stable Diffusion**

* Stable Diffusion 3.5 is then loaded through the Hugging Face Diffusers library.
* Text prompts are converted into embeddings so they can be used as numerical conditioning by Stable Diffusion.
* Stable Diffusion is frozen during training in order to reduce computational costs and use it as a fixed pretrained backbone rather than performing full fine-tuning. Only the Adapter is trained.

### **Adapter**

* Before training, a custom Adapter is created. Its role is to generate additional conditioning tokens from the selected embedding. These tokens are the core of the project because they are designed to provide Stable Diffusion with extra structural information.
* The initial version of the Adapter is saved. This is not essential, but it can be useful for reproducibility, comparing the initial and trained states, or restarting training from the same configuration.
* A custom Dataset is then created using all selected urban images together with the chosen embedding type: semantic segmentation, GNN, or the fusion of both. The same image preprocessing is applied to all images to make training more consistent.
* The dataset is shuffled and divided into multiple batches of size `BATCH_SIZE`.
* The Adapter is then trained for a predefined number of `STEPS`, where each step corresponds to one batch and one parameter update.

## **Dataset**

* Panoptic Scene Graph
* COCO

## **Training**

For each batch:
* Retrieve the images and their associated embeddings.
* Encode the real images into latent representations using the VAE.
* Generate random noise for each image and compute the timestep, the noisy latent, and the flow-matching target.
* Use the positive text embeddings as textual conditioning during Adapter training.
* Generate additional conditioning tokens with the Adapter and concatenate them with the positive prompt embeddings.
* Ask the frozen Stable Diffusion Transformer to predict the expected flow direction from the noisy latent and the enriched conditioning.
* Compute the loss by comparing the model prediction with the expected target.
* Backpropagate the loss and update the Adapter parameters so that it can progressively generate more useful conditioning tokens.

## **Experiments**

* Experiment A - Semantic segmentation embedding
* Experiment B - Scene-graph GNN embedding
* Experiment C - Fused representation

## **Limitations**

1. The GNN, semantic segmentation encoder, and fusion block are not trained before the embeddings are generated. Their weights therefore remain randomly initialized. The Adapter is trained afterwards using these embeddings, but the upstream representations themselves were not learned to capture meaningful scene semantics. A better approach would be to train or pretrain these components before using their embeddings.
2. Each image is also associated with the same textual prompt during Adapter training. As a result, the Adapter does not learn to interact with a wide variety of user requests. For example, if a user later asks for a red car, the training process never explicitly teaches the Adapter how different textual instructions should interact with the structural embeddings.

## **Technologies**

* Python
* PyTorch
* Hugging Face
* Stable Diffusion 3.5 Medium
* Weights & Biases
* torchvision
* NetworkX
* PyVis
* Panoptic Scene Graph
* NumPy
* Pandas
* Matplotlib
* Pillow
* CUDA / GPU acceleration