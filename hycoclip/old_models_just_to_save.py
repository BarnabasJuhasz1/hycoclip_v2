# for printing
import json
class HyCoCLIP_V2(HyCoCLIP):
    """
    Our HyCoCLIP_V2 model, that modifies HyCoCLIP to embed images, texts,
    their localized box and hierarchy information in a hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_hierarchies: bool = True,
        cont_weights: tuple[float, float, float, float] = [1, .75, .5, .25],
    ):
        """
        Un-documented args are same as `HyCoCLIP`.

        Args:
            use_hierarchies: Whether to use caption hierarchies for training.
            cont_weights: Hyperparameters for hierarchical contrastive loss weights.

        """
        super().__init__(visual, textual, embed_dim, curv_init, learn_curv, entail_weight, use_boxes, pixel_mean, pixel_std)
        self.cont_weights = cont_weights
        assert use_hierarchies, "HyCoCLIP_V2 requires caption hierarchies to function."

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
        hierarchy_tokens: list[list[torch.Tensor]], pairwise_scores: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            hierarchy_tokens: list of list of tensors, each containing hierarchy tokens
                for each phrase in the hierarchy.
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)
        # print(f"image_feats shape: {image_feats.shape}")
        # print(f"text_feats shape: {text_feats.shape}")


        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        hierarchy_feats_0 = self.encode_text([hier[0] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_1 = self.encode_text([hier[1] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_2 = self.encode_text([hier[2] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_3 = self.encode_text([hier[3] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_4 = self.encode_text([hier[4] for hier in hierarchy_tokens], project=True)



        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)
        all_box_text_feats = dist.gather_across_processes(box_text_feats)
        all_box_image_feats = dist.gather_across_processes(box_image_feats)
        # all_hier_feats = dist.gather_across_processes(hierarchy_feats)

        all_hier_feats_0 = dist.gather_across_processes(hierarchy_feats_0)
        all_hier_feats_1 = dist.gather_across_processes(hierarchy_feats_1)
        all_hier_feats_2 = dist.gather_across_processes(hierarchy_feats_2)
        all_hier_feats_3 = dist.gather_across_processes(hierarchy_feats_3)
        all_hier_feats_4 = dist.gather_across_processes(hierarchy_feats_4)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)
        all_box_text_feats = torch.cat(all_box_text_feats, dim=0)
        all_box_image_feats = torch.cat(all_box_image_feats, dim=0)
        # shape: (batch_size * world_size, hierarchy_size(=5), embed_dim)
        # all_hier_feats = torch.cat(all_hier_feats, dim=0)
        all_hier_feats_0 = torch.cat(all_hier_feats_0, dim=0)
        all_hier_feats_1 = torch.cat(all_hier_feats_1, dim=0)
        all_hier_feats_2 = torch.cat(all_hier_feats_2, dim=0)
        all_hier_feats_3 = torch.cat(all_hier_feats_3, dim=0)
        all_hier_feats_4 = torch.cat(all_hier_feats_4, dim=0)


        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
            # print("IMAGE LOGITS: ", image_logits)
            # print("TEXT LOGITS: ", text_logits)
            box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
            box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

            # # hierarchy_feats_0 = box_text_feats = feats for the sub_caption
            # hier_logits_1 = -L.pairwise_dist(hierarchy_feats_1, all_box_image_feats, _curv)
            # hier_logits_2 = -L.pairwise_dist(hierarchy_feats_2, all_box_image_feats, _curv)
            # hier_logits_3 = -L.pairwise_dist(hierarchy_feats_3, all_box_image_feats, _curv)
            # #  hierarchy_feats_4 = feats for the most general term like "object" or "animal"
            # hier_logits_4 = -L.pairwise_dist(hierarchy_feats_4, all_box_image_feats, _curv)


            # using hyperbolic distance
            hier_logits_1 = L.elementwise_dist(hierarchy_feats_0, hierarchy_feats_1, _curv)
            hier_logits_2 = L.elementwise_dist(hierarchy_feats_1, hierarchy_feats_2, _curv)
            hier_logits_3 = L.elementwise_dist(hierarchy_feats_2, hierarchy_feats_3, _curv)
            hier_logits_4 = L.elementwise_dist(hierarchy_feats_3, hierarchy_feats_4, _curv)

            # # mean 
            mean_elementwise_dist = 0.25*((hier_logits_1**2).mean() + (hier_logits_2**2).mean() + (hier_logits_3**2).mean() + (hier_logits_4**2).mean())


            hier_logits_easy = -L.pairwise_dist(hierarchy_feats_0, all_hier_feats_4, _curv)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()


            contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)
            )

            contrastive_loss_easy = nn.functional.cross_entropy(_scale * hier_logits_easy, targets)

            # original: 0.25
            new_contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)

                # ReCon5
                # + self.cont_weights[0] * nn.functional.cross_entropy(_scale * hier_logits_1, targets)
                # + self.cont_weights[1] * nn.functional.cross_entropy(_scale * hier_logits_2, targets)
                # + self.cont_weights[2] * nn.functional.cross_entropy(_scale * hier_logits_3, targets)
                # + self.cont_weights[3] * nn.functional.cross_entropy(_scale * hier_logits_4, targets)

                # ReCon6
                # + nn.functional.cross_entropy(_scale * hier_logits_1, targets)
                # + nn.functional.cross_entropy(_scale * hier_logits_2, targets)
                # + nn.functional.cross_entropy(_scale * hier_logits_3, targets)
                # + nn.functional.cross_entropy(_scale * hier_logits_4, targets)

                # + contrastive_loss_easy
            )

            # print("text_feats", text_feats.shape, text_feats.isnan().any(), text_feats.isinf().any())
            # print("image_feats", image_feats.shape, image_feats.isnan().any(), image_feats.isinf().any())
            # print("_curv", _curv)

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
            _box_aperture = L.half_aperture(box_text_feats, _curv)

            _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
            _box_image_aperture = L.half_aperture(box_image_feats, _curv)

            _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
            _box_text_aperture = L.half_aperture(box_text_feats, _curv)


            # hier_angle_0 = L.oxy_angle(hierarchy_feats_0, image_feats, _curv)
            # hier_angle_1 = L.oxy_angle(hierarchy_feats_1, box_text_feats, _curv)
            # hier_angle_2 = L.oxy_angle(hierarchy_feats_2, box_text_feats, _curv)
            # hier_angle_3 = L.oxy_angle(hierarchy_feats_3, box_text_feats, _curv)
            # hier_angle_4 = L.oxy_angle(hierarchy_feats_4, box_text_feats, _curv)


            hier_chain_angle_1 = L.oxy_angle(hierarchy_feats_1, hierarchy_feats_0, _curv)
            hier_chain_angle_2 = L.oxy_angle(hierarchy_feats_2, hierarchy_feats_1, _curv)
            hier_chain_angle_3 = L.oxy_angle(hierarchy_feats_3, hierarchy_feats_2, _curv)
            hier_chain_angle_4 = L.oxy_angle(hierarchy_feats_4, hierarchy_feats_3, _curv)


            hier_angle_easy = L.oxy_angle(hierarchy_feats_4, hierarchy_feats_0, _curv)


            # hier_aperture_0 = L.half_aperture(hierarchy_feats_0, _curv)
            hier_aperture_1 = L.half_aperture(hierarchy_feats_1, _curv)
            hier_aperture_2 = L.half_aperture(hierarchy_feats_2, _curv)
            hier_aperture_3 = L.half_aperture(hierarchy_feats_3, _curv)
            hier_aperture_4 = L.half_aperture(hierarchy_feats_4, _curv)

            hier_aperture_easy = L.half_aperture(hierarchy_feats_4, _curv)

            # Hyperparameters for apertures
            _global_aperture_thresh = 1   # inter-modal --> 1 ? 
            _local_aperture_thresh = 1.2    # intra-modal --> remains 1.2

            text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
            box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
            cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
            cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()


            pairwise_scores_masked = torch.where(pairwise_scores < 0.6, pairwise_scores, 1)
            # pairwise_scores_0 = [pairwise_scores[0] if pairwise_scores[0] < 0.6 else 1 for scores in pairwise_scores]
            # pairwise_scores_1 = [pairwise_scores[1] if pairwise_scores[1] < 0.6 else 1 for scores in pairwise_scores]
            # pairwise_scores_2 = [pairwise_scores[2] if pairwise_scores[2] < 0.6 else 1 for scores in pairwise_scores]
            # pairwise_scores_3 = [pairwise_scores[3] if pairwise_scores[3] < 0.6 else 1 for scores in pairwise_scores]


            # repair experiment: 0.7 * [1, 2] => [0.7, 1.4] but only scores < 0.6
            # repair experiment: hier_entailment_loss_1_glob = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1 * torch.clamp((2-pairwise_scores[:,0]), min=0.1), min=0).mean()

            # middle-ground experiment: 0.7 * [0.5, 1.5] => [0.35, 1.05]
            # middle-ground experiment: hier_entailment_loss_1_glob = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1 * (1.5-pairwise_scores[:,0]), min=0).mean()
            # strict experiment: 1-pairwise_scores => [0, 1]
            # strict experiment: hier_entailment_loss_1_glob = torch.clamp(hier_chain_angle_1 - hier_aperture_1 * torch.clamp((1-pairwise_scores[:,0]), min=0.1), min=0).mean()

            # less strict experiment: 0.7 * [1, 2] => [0.7, 1.4] but scores can be anything
            # less strict experiment: hier_entailment_loss_1_glob = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1 * torch.clamp((2-pairwise_scores[:,0]), min=0.1), min=0).mean()

            # entailment chain with half aperture and pairwise scores
            # hier_entailment_loss_1_glob = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1 * torch.clamp((2-pairwise_scores_masked[:,0]), min=0.1), min=0).mean()
            # hier_entailment_loss_2_glob = torch.clamp(hier_chain_angle_2 - _global_aperture_thresh * hier_aperture_2 * torch.clamp((2-pairwise_scores_masked[:,1]), min=0.1), min=0).mean()
            # hier_entailment_loss_3_glob = torch.clamp(hier_chain_angle_3 - _global_aperture_thresh * hier_aperture_3 * torch.clamp((2-pairwise_scores_masked[:,2]), min=0.1), min=0).mean()
            # hier_entailment_loss_4_glob = torch.clamp(hier_chain_angle_4 - _global_aperture_thresh * hier_aperture_4 * torch.clamp((2-pairwise_scores_masked[:,3]), min=0.1), min=0).mean()

            # ReCos experiment: entailment chain with arccosine, half aperture removed
            # hier_entailment_loss_1_glob = torch.clamp(hier_chain_angle_1 - torch.acos(pairwise_scores[:,0]), min=0).mean()
            # hier_entailment_loss_2_glob = torch.clamp(hier_chain_angle_2 - torch.acos(pairwise_scores[:,1]), min=0).mean()
            # hier_entailment_loss_3_glob = torch.clamp(hier_chain_angle_3 - torch.acos(pairwise_scores[:,2]), min=0).mean()
            # hier_entailment_loss_4_glob = torch.clamp(hier_chain_angle_4 - torch.acos(pairwise_scores[:,3]), min=0).mean()


            # ReWeight: entailment chain basic
            # hier_entailment_loss_1_glob = torch.clamp(hier_chain_angle_1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()
            # hier_entailment_loss_2_glob = torch.clamp(hier_chain_angle_2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()
            # hier_entailment_loss_3_glob = torch.clamp(hier_chain_angle_3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()
            # hier_entailment_loss_4_glob = torch.clamp(hier_chain_angle_4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()


            hier_entailment_loss_easy_global = torch.clamp(hier_angle_easy - _global_aperture_thresh * hier_aperture_easy, min=0).mean()
            hier_entailment_loss_easy_local = torch.clamp(hier_angle_easy - _local_aperture_thresh * hier_aperture_easy, min=0).mean()

            entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss
            )

            #measuring entailment of general term H3 with text-box H0
            # hier_angle_H1 = L.oxy_angle(hierarchy_feats_1, hierarchy_feats_0, _curv)
            # hier_entailment_loss_H1_global = torch.clamp(hier_angle_H1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()

            # #middle difficulty, measuring entailment of middle element H2 with text-box H0
            # hier_angle_H2 = L.oxy_angle(hierarchy_feats_2, hierarchy_feats_0, _curv)
            # hier_entailment_loss_H2_global = torch.clamp(hier_angle_H2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()

            # #measuring entailment of general term H3 with text-box H0
            # hier_angle_H3 = L.oxy_angle(hierarchy_feats_3, hierarchy_feats_0, _curv)
            # hier_entailment_loss_H3_global = torch.clamp(hier_angle_H3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()

            # #easy difficulty, measuing entailment of last element H4 (general) with text-box H0
            # hier_angle_H4 = L.oxy_angle(hierarchy_feats_4, hierarchy_feats_0, _curv)
            # hier_entailment_loss_H4_global = torch.clamp(hier_angle_H4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()
            # hier_entailment_loss_easy_local = torch.clamp(hier_angle_H4 - _local_aperture_thresh * hier_aperture_4, min=0).mean()

            # original: 0.5
            new_entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss

                # hierarchical entailment chain
                # + hier_entailment_loss_1_glob
                # + hier_entailment_loss_2_glob
                # + hier_entailment_loss_3_glob
                # + hier_entailment_loss_4_glob

                # + hier_entailment_loss_easy_local

                # ReWeight with individual entailment loss being the original
                # + (pairwise_scores[0] * hier_entailment_loss_1_glob).mean()
                # + (pairwise_scores[1] * hier_entailment_loss_2_glob).mean()
                # + (pairwise_scores[2] * hier_entailment_loss_3_glob).mean()
                # + (pairwise_scores[3] * hier_entailment_loss_4_glob).mean()

            )


            # hierarchical_features = [
            #     hierarchy_feats_1,
            #     hierarchy_feats_2,
            #     hierarchy_feats_3,
            #     hierarchy_feats_4
            # ]

            # hierarchical_apertures = [
            #     hier_aperture_1,
            #     hier_aperture_2,
            #     hier_aperture_3,
            #     hier_aperture_4
            # ]

            # hierarchical_entailment_global_losses = [
            #     hier_entailment_loss_H1_global,
            #     hier_entailment_loss_H2_global,
            #     hier_entailment_loss_H3_global,
            #     hier_entailment_loss_H4_global
            # ]

            # RS1 = False
            # if(RS1):
            #     index1 = random.randint(1, 4)
            #     #print(f"With RS1, we rolled {index1}!")
            #     # H_random > Tbox
            #     new_entailment_loss += (0.5 * hierarchical_entailment_global_losses[index1-1]) 

            loss = contrastive_loss
            new_loss = new_contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss
                new_loss = new_loss + self.entail_weight * new_entailment_loss #+ 0.2 * mean_elementwise_dist

            # print("angle_easy:", hier_angle_easy.mean().item())
            # print("thresh × aperture_easy:", (_local_aperture_thresh * hier_aperture_easy).mean().item())

        returnDict = {
            "loss": new_loss,
            "logging": {
                "loss": new_loss,
                "contrastive_loss": new_contrastive_loss,
                "entailment_loss": new_entailment_loss,
                "old_loss": loss,
                "old_contrastive_loss": contrastive_loss,
                "old_entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
                # "hier_entailment_loss_1 global": hier_entailment_loss_1_glob,
                # "hier_entailment_loss_2 global": hier_entailment_loss_2_glob,
                # "hier_entailment_loss_3 global": hier_entailment_loss_3_glob,
                # "hier_entailment_loss_4 global": hier_entailment_loss_4_glob,
            }
        }

        # for name, value in returnDict["logging"].items():
        #     if torch.isnan(value).any():
        #         print(f"⚠️ NaN detected in {name}!")
        # print(json.dumps(dict_for_print(returnDict), indent=4))
        # print()
        return returnDict


def tensor_to_value(x):
    if isinstance(x, torch.Tensor):
        return x.item() if x.ndim == 0 else x.tolist()
    return x

def dict_for_print(d):
    if isinstance(d, dict):
        return {k: dict_for_print(v) for k, v in d.items()}
    return tensor_to_value(d)

class HyCoCLIP_V2_EASY(HyCoCLIP):
    """
    Our HyCoCLIP_V2 model, that modifies HyCoCLIP to embed images, texts,
    their localized box and hierarchy information in a hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_hierarchies: bool = True,
        cont_weights: tuple[float, float, float, float] = [1, .75, .5, .25],
    ):
        """
        Un-documented args are same as `HyCoCLIP`.

        Args:
            use_hierarchies: Whether to use caption hierarchies for training.
            cont_weights: Hyperparameters for hierarchical contrastive loss weights.

        """
        super().__init__(visual, textual, embed_dim, curv_init, learn_curv, entail_weight, use_boxes, pixel_mean, pixel_std)
        self.cont_weights = cont_weights
        assert use_hierarchies, "HyCoCLIP_V2 requires caption hierarchies to function."

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
        hierarchy_tokens: list[list[torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            hierarchy_tokens: list of list of tensors, each containing hierarchy tokens
                for each phrase in the hierarchy.
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        hierarchy_feats_0 = self.encode_text([hier[0] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_1 = self.encode_text([hier[1] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_2 = self.encode_text([hier[2] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_3 = self.encode_text([hier[3] for hier in hierarchy_tokens], project=True)
        hierarchy_feats_4 = self.encode_text([hier[4] for hier in hierarchy_tokens], project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        all_hier_feats_0 = dist.gather_across_processes(hierarchy_feats_0)
        all_hier_feats_1 = dist.gather_across_processes(hierarchy_feats_1)
        all_hier_feats_2 = dist.gather_across_processes(hierarchy_feats_2)
        all_hier_feats_3 = dist.gather_across_processes(hierarchy_feats_3)
        all_hier_feats_4 = dist.gather_across_processes(hierarchy_feats_4)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)
        # shape: (batch_size * world_size, hierarchy_size(=5), embed_dim)
        all_hier_feats_0 = torch.cat(all_hier_feats_0, dim=0)
        all_hier_feats_1 = torch.cat(all_hier_feats_1, dim=0)
        all_hier_feats_2 = torch.cat(all_hier_feats_2, dim=0)
        all_hier_feats_3 = torch.cat(all_hier_feats_3, dim=0)
        all_hier_feats_4 = torch.cat(all_hier_feats_4, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
            box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
            box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

            # hierarchy_feats_0 = box_text_feats = feats for the sub_caption
            hier_logits_0 = -L.pairwise_dist(hierarchy_feats_0, all_box_image_feats, _curv)
            hier_logits_1 = -L.pairwise_dist(hierarchy_feats_1, all_box_image_feats, _curv)
            hier_logits_2 = -L.pairwise_dist(hierarchy_feats_2, all_box_image_feats, _curv)
            #  hierarchy_feats_4 = feats for the most general term like "object" or "animal"
            hier_logits_3 = -L.pairwise_dist(hierarchy_feats_3, all_box_image_feats, _curv)


            hier_logits_easy = -L.pairwise_dist(hierarchy_feats_0, all_box_image_feats, _curv)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)
            )

            contrastive_loss_easy = nn.functional.cross_entropy(_scale * hier_logits_easy, targets)
            new_contrastive_loss = 0.125 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)

                # ReCon6
                + nn.functional.cross_entropy(_scale * hier_logits_1, targets)
                + nn.functional.cross_entropy(_scale * hier_logits_2, targets)
                + nn.functional.cross_entropy(_scale * hier_logits_3, targets)
                + nn.functional.cross_entropy(_scale * hier_logits_4, targets)

            )

            # print("text_feats", text_feats.shape, text_feats.isnan().any(), text_feats.isinf().any())
            # print("image_feats", image_feats.shape, image_feats.isnan().any(), image_feats.isinf().any())
            # print("_curv", _curv)

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
            _box_aperture = L.half_aperture(box_text_feats, _curv)

            _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
            _box_image_aperture = L.half_aperture(box_image_feats, _curv)

            _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
            _box_text_aperture = L.half_aperture(box_text_feats, _curv)


            # hier_angle_0 = L.oxy_angle(hierarchy_feats_0, image_feats, _curv)
            hier_angle_1 = L.oxy_angle(hierarchy_feats_1, hierarchy_feats_0, _curv)
            hier_angle_2 = L.oxy_angle(hierarchy_feats_2, hierarchy_feats_1, _curv)
            hier_angle_3 = L.oxy_angle(hierarchy_feats_3, hierarchy_feats_2, _curv)
            hier_angle_4 = L.oxy_angle(hierarchy_feats_4, hierarchy_feats_3, _curv)


            # hier_aperture_0 = L.half_aperture(hierarchy_feats_0, _curv)
            hier_aperture_1 = L.half_aperture(hierarchy_feats_1, _curv)
            hier_aperture_2 = L.half_aperture(hierarchy_feats_2, _curv)
            hier_aperture_3 = L.half_aperture(hierarchy_feats_3, _curv)
            hier_aperture_4 = L.half_aperture(hierarchy_feats_4, _curv)

            # Hyperparameters for apertures
            _global_aperture_thresh = 0.7   # inter-modal
            _local_aperture_thresh = 1.2    # intra-modal

            text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
            box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
            cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
            cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()
            
            hier_entailment_loss_1 = torch.clamp(hier_angle_1 - _local_aperture_thresh * hier_aperture_1, min=0).mean()
            hier_entailment_loss_2 = torch.clamp(hier_angle_2 - _local_aperture_thresh * hier_aperture_2, min=0).mean()
            hier_entailment_loss_3 = torch.clamp(hier_angle_3 - _local_aperture_thresh * hier_aperture_3, min=0).mean()
            hier_entailment_loss_4 = torch.clamp(hier_angle_4 - _local_aperture_thresh * hier_aperture_4, min=0).mean()

            hier_entailment_loss_1_glob = torch.clamp(hier_angle_1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()
            hier_entailment_loss_2_glob = torch.clamp(hier_angle_2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()
            hier_entailment_loss_3_glob = torch.clamp(hier_angle_3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()
            hier_entailment_loss_4_glob = torch.clamp(hier_angle_4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()


            #measuring entailment of general term H3 with text-box H0
            hier_angle_H1 = L.oxy_angle(hierarchy_feats_1, hierarchy_feats_0, _curv)
            hier_entailment_loss_H1_global = torch.clamp(hier_angle_H1 - _global_aperture_thresh * hier_aperture_1, min=0).mean()

            #middle difficulty, measuring entailment of middle element H2 with text-box H0
            hier_angle_H2 = L.oxy_angle(hierarchy_feats_2, hierarchy_feats_0, _curv)
            hier_entailment_loss_H2_global = torch.clamp(hier_angle_H2 - _global_aperture_thresh * hier_aperture_2, min=0).mean()

            #measuring entailment of general term H3 with text-box H0
            hier_angle_H3 = L.oxy_angle(hierarchy_feats_3, hierarchy_feats_0, _curv)
            hier_entailment_loss_H3_global = torch.clamp(hier_angle_H3 - _global_aperture_thresh * hier_aperture_3, min=0).mean()

            #easy difficulty, measuing entailment of last element H4 (general) with text-box H0
            hier_angle_H4 = L.oxy_angle(hierarchy_feats_4, hierarchy_feats_0, _curv)
            hier_entailment_loss_H4_global = torch.clamp(hier_angle_H4 - _global_aperture_thresh * hier_aperture_4, min=0).mean()
            hier_entailment_loss_easy_local = torch.clamp(hier_angle_H4 - _local_aperture_thresh * hier_aperture_4, min=0).mean()

            entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss
            )

            new_entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss

                # + hier_entailment_loss_H4_global
                # + hier_entailment_loss_H2_global
                
                # + hier_entailment_loss_1
                # + hier_entailment_loss_2
                # + hier_entailment_loss_3
                # + hier_entailment_loss_4

                # + hier_entailment_loss_1_glob
                # + hier_entailment_loss_2_glob
                # + hier_entailment_loss_3_glob
                # + hier_entailment_loss_4_glob

            )

            hierarchical_features = [
                hierarchy_feats_1,
                hierarchy_feats_2,
                hierarchy_feats_3,
                hierarchy_feats_4
            ]

            hierarchical_apertures = [
                hier_aperture_1,
                hier_aperture_2,
                hier_aperture_3,
                hier_aperture_4
            ]

            hierarchical_entailment_global_losses = [
                hier_entailment_loss_H1_global,
                hier_entailment_loss_H2_global,
                hier_entailment_loss_H3_global,
                hier_entailment_loss_H4_global
            ]

            RS1 = True
            RS2 = False
            RS3 = False
            if(RS1):
                index1 = random.randint(1, 4)
                #print(f"With RS1, we rolled {index1}!")
                # H_random > Tbox
                new_entailment_loss += (0.5 * hierarchical_entailment_global_losses[index1-1]) 
            elif (RS2):
                index1 = random.randint(1, 2)
                index2 = random.randint(3, 4)
                #print(f"With RS2, we rolled {index1} and {index2}!")
                # H_random1 > Tbox and H_random2 > Tbox
                new_entailment_loss += (0.5 * hierarchical_entailment_global_losses[index1-1]) 
                new_entailment_loss += (0.5 * hierarchical_entailment_global_losses[index2-1]) 
            elif (RS3):
                index1 = random.randint(1, 2)
                index2 = random.randint(3, 4)
                #print(f"With RS3, we rolled {index1} and {index2}!")
                # H_random_1 > H_random2 > Tbox  (where H_random_1 is the larger index because it is more general)
                index_larger = max(index1, index2)
                index_smaller = min(index1, index2)

                random_h_feats_larger = hierarchical_features[index_larger-1]
                random_h_feats_smaller = hierarchical_features[index_smaller-1]

                random_h_aperture_larger = hierarchical_apertures[index_larger-1]
                random_h_aperture_smaller = hierarchical_apertures[index_smaller-1]

                # H_random_1 > H_random2
                hier_angle_larger = L.oxy_angle(random_h_feats_larger, random_h_feats_smaller, _curv)
                hier_entailment_loss_larger = torch.clamp(hier_angle_larger - _global_aperture_thresh * random_h_aperture_larger, min=0).mean()
                new_entailment_loss += (0.5 * hier_entailment_loss_larger) 
                
                # H_random_2 > Tbox
                hier_angle_smaller = L.oxy_angle(random_h_feats_smaller, hierarchy_feats_0, _curv)
                hier_entailment_loss_smaller = torch.clamp(hier_angle_smaller - _global_aperture_thresh * random_h_aperture_smaller, min=0).mean()
                new_entailment_loss += (0.5 * hier_entailment_loss_smaller) 


            loss = contrastive_loss
            new_loss = new_contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss
                new_loss = new_loss + self.entail_weight * new_entailment_loss

            # print("angle_easy:", hier_angle_easy.mean().item())
            # print("thresh × aperture_easy:", (_local_aperture_thresh * hier_aperture_easy).mean().item())

        returnDict = {
            "loss": new_loss,
            "logging": {
                "loss": new_loss,
                "contrastive_loss": new_contrastive_loss,
                "entailment_loss": new_entailment_loss,
                "hier_entailment_loss_1": hier_entailment_loss_1,
                "hier_entailment_loss_2": hier_entailment_loss_2,
                "hier_entailment_loss_3": hier_entailment_loss_3,
                "hier_entailment_loss_4": hier_entailment_loss_4,
                "old_loss": loss,
                "old_contrastive_loss": contrastive_loss,
                "old_entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
                "easy contrastive loss": contrastive_loss_easy,
                "easy entailment loss local": hier_entailment_loss_easy_local,
                "hier_entailment_loss_H4_global": hier_entailment_loss_H4_global,
                "hier_entailment_loss_H2_global": hier_entailment_loss_H2_global,
                "hier_entailment_loss_1 global": hier_entailment_loss_1_glob,
                "hier_entailment_loss_2 global": hier_entailment_loss_2_glob,
                "hier_entailment_loss_3 global": hier_entailment_loss_3_glob,
                "hier_entailment_loss_4 global": hier_entailment_loss_4_glob,
            }
        }

        # for name, value in returnDict["logging"].items():
        #     if torch.isnan(value).any():
        #         print(f"⚠️ NaN detected in {name}!")
        # print(json.dumps(dict_for_print(returnDict), indent=4))
        # print()
        return returnDict



