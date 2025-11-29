- Change text so it does not say the official Pytorch repo
- Change beam size to 3

The following function performs the caption generation using the pretrained SAT model. For preprocessing, the input image is resized to 256×256, converted to RGB if necessary, scaled to the [0,1] range, and normalized with ImageNet statistics so it matches the ResNet-101 encoder described above. The encoded image features are then passed to the SAT decoder, which works as outlined above: The decoder applies the learned attention mechanism over the 14×14 spatial feature grid and generates one word at a time using the LSTM.

Caption generation is implemented with beam search, where the function keeps the top-k partial sequences at each timestep and updates them based on the decoder’s log-probabilities. As in the standard SAT approach, each decoding step also produces an attention map showing which spatial regions contributed to the predicted word, and these maps are stored throughout the sequence.

The function returns the highest-scoring caption together with the corresponding attention weights and the token list. Since it directly follows the SAT pipeline in the AICAttack paper, this implementation remains compatible with existing sgrvinod SAT checkpoints and behaves consistently with the original model design.

---

(The attention comparison highlights a key architectural difference between SAT and ViT-GPT2. SAT shows sharply localized attention peaks due to its CNN-based feature grid and stepwise soft-attention mechanism, while ViT-GPT2 distributes attention across many patches, reflecting the transformer’s global receptive field.) Based on these visualizations, we assume that the pixel-level perturbations used in AICAttack will not transfer directly. Targeting a small set of high-attention regions works well for SAT, but ViT-GPT2’s more diffuse attention most likely makes localized perturbations less disruptive. Therefore, it is likely that we require larger perturbation regions to achieve similar results as seen in the AIC Attack paper.