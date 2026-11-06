# Literature review for Study 2

**Date opened.** 6 November 2026.

**Scope.** Cross-prompt adversarial transfer, context-dependent robustness of vision-language models, gradient alignment as a predictor of transfer, multimodal gradient conflict, counterfactual visual grounding, and causal interventions. The review separates results stated in sources that were actually opened from the estimate this proposal would add.

**Method.** Papers were located by search, then read from their HTML or PDF. A search snippet was not treated as a source. Claims below track the pages that were opened. Where a paper asserts novelty, that assertion is reported as the paper's claim.

**What was not opened, and is therefore not evidence here.** The full text of "Say What I Want!" (the IEEE prompt-agnostic attack paper that appeared in search), Moosavi-Dezfooli et al. on universal perturbations, Wang et al. 2020 on multi-modal training difficulty, Meng et al. on causal tracing in language models, the Qwen3 technical report, Xie et al. on chain-of-attack transfer, and any license file for a pre-rendered CLEVR image archive. Palit et al. describe their debt to Meng et al.; that description is not a substitute for opening Meng et al.

Study 1 is a primary input and is reviewed in §7. Its measurements are observations about one frozen protocol, not general laws.

## 1. Cross-prompt adversarial transfer

### 1.1 CroPA and the Multi-P baseline

Luo, Gu, Liu, and Torr, in the paper whose ICLR 2024 proceedings PDF and arXiv HTML were both opened (`arxiv.org/abs/2403.09766`), ask whether one adversarial image can mislead a vision-language model under many prompts. They call that property cross-prompt adversarial transferability.

What they report:

- An image optimized against a single fixed prompt (Single-P) has highly limited cross-prompt transfer.
- Optimizing the image against several fixed prompts (Multi-P) raises transfer. In their targeted experiments the gain is clearest when the prompt count goes from one to five. From about ten prompts upward, further prompts add little, especially for the baseline that does not learn a prompt perturbation.
- CroPA keeps a set of prompt embeddings in the loop and updates them by gradient ascent on the language-modeling loss, while the image perturbation is updated by gradient descent. The optimizer is projected gradient descent in L∞. At test time the object of interest is the image. The learned prompt perturbation is an optimization device.
- The image objective pushes the model toward a chosen target string. A common target in the paper is "unknown," chosen to avoid a high-frequency answer. They also report a non-targeted setting in which success means any change of prediction, and CroPA remains ahead of the baseline there.
- Tasks are image classification, captioning, and VQA. Models include Flamingo, BLIP-2, and InstructBLIP.
- Transfer across images, or across models with very different architectures, is weak in their experiments. They describe an overall attack success near 10% for the InstructBLIP and BLIP-2 pair, and near zero for more distant pairs. They treat cross-image and cross-model methods as orthogonal add-ons.

The prompts in that study are different task instructions, including long lists of VQA questions, not a matched pair of captions wrapped around one fixed question. The attack is aimed at a target string, not at raising the negative log probability of a scene's true answer. Those two differences are the gap a leakage-controlled scene study can sit in. They do not make Multi-P unpublished. Multi-context PGD, in the sense of projected gradient steps on an image loss averaged over several fixed texts, is Multi-P.

### 1.2 Gradient regularization across prompts

The OpenReview PDF `I05Z6KjQ9K` is an anonymous manuscript whose banner reads "Under review as a conference paper at ICLR 2025." The title is "Gradient Regularization-based Cross-Prompt Attacks on Vision Language Models." Because it is anonymous and under review, its tables are treated as claims in a preprint, not as a confirmed venue result.

The manuscript defines Single-P and Multi-P in the same way as Luo et al., cites them, and says that optimizing against many prompts at once is severely non-stationary and overfits individual prompts. The proposed remedy, GrCPA, clips back-propagated gradients inside attention and MLP blocks. The authors describe the method as orthogonal to CroPA. Evaluations named in the paper include OpenFlamingo-9B, BLIP-2, LLaVA-1.5-7B, and InstructBLIP, on MS-COCO images with VQA, classification, and caption prompts, again toward chosen target strings.

The manuscript uses gradients to build a more transferable image. It does not report a scene-level regression of transfer success on the sign alignment of two context-conditional image gradients.

### 1.3 A reproducibility preprint, and a mutual-information attacker

Mittal, Pandey, Tiwari, Jindal, and Swain (`arxiv.org/html/2506.22982v1`) report that they reproduced CroPA's main comparisons with Single-P and Multi-P, and then added initialization, universal-perturbation, and attention-value losses. This is a preprint, not a second venue publication. It is used here only as an independent group's statement that the Single-P / Multi-P / CroPA ordering was reproducible in their runs. Their new losses are additional attack machinery, not a test of predictive alignment.

Cai, Liu, Qu, Fang, Dong, Tang, Zhou, Sun, and Hu, in the NeurIPS 2025 PDF "Towards Building Model/Prompt-Transferable Attackers against Large Vision-Language Models," start from the observation that image attacks overfit a model and a prompt. They write that the prompt-agnostic attacks already in the literature rely on multi-prompt joint training and still leave cross-model transfer open. Their method estimates mutual information between the output and a split of the image into a benign part and a perturbation, then pushes the output to depend more on the perturbation. Success is measured by semantic similarity to a target string such as "I am sorry," and by exact match, on LLaVA-1.5, MiniGPT-4, BLIP-2, and InstructBLIP. The axis is again "make one image follow the attacker across prompts and models," not "does gradient agreement forecast error on a fixed question."

### 1.4 Established versus open

Established in these sources: a visual adversarial example is prompt-dependent; optimizing it on more than one fixed prompt is a known baseline; learnable prompt perturbations and gradient clipping and mutual-information penalties are published or preprinted attempts to push transfer further; those attempts mostly target a chosen string on natural or VQA images.

Open relative to these sources: whether, for one model and one question, the sign alignment of \(\nabla_x (-\log P(y \mid x, c))\) under two contexts that do not contain \(y\) predicts strict-error transfer of PGD on a counterfactual scene. That open estimate is H\* in the proposal. Absence from this reading list is not a proof that no such estimate exists.

## 2. Context-dependent robustness

Cui, Aparcedo, Jang, and Lim, CVPR 2024, study image adversarial attacks on LLaVA, BLIP-2, and InstructBLIP for classification, caption retrieval, and VQA. The PDF opened from the NSF repository states the venue. Their attacks are often computed against the visual encoder (CLIP or EVA-CLIP), not always end-to-end against the answer loss. Findings they state:

- The multimodal models are not robust to visual adversarial inputs in general. Caption retrieval collapses.
- VQA is more resilient than caption retrieval in their tables. They conjecture that a question is less affected when it asks about something other than the content the attack targeted.
- Additional textual context mitigates the visual attack. The abstract's summary figure for ScienceQA is an 8.10% drop against a 99.73% drop for the visual counterparts. The body table shows that the VQA drop depends on the model and the attack column; the abstract's single percentage is their summary, not a constant.

"Context" in that paper means task text and ScienceQA question context. It does not mean a false caption wrapped around an unchanged question. The mitigation finding is real for the prompts they added. It does not say that every extra sentence protects the model, and it does not say that a sentence which restates the answer is a visual defense.

Zhao, Pang, Du, Yang, Li, Cheung, and Lin, NeurIPS 2023, evaluate black-box targeted attacks on MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, and Img2Prompt. Surrogates are CLIP and BLIP. The default instruction they fix is "what is the content of this image?". Transfer across models is the axis. The question is not varied as an experimental factor. They also show a qualitative multi-turn effect on MiniGPT-4. That is evidence that an image attack can keep steering a conversation, not evidence about matched non-revealing contexts.

Schlarmann and Hein, in the ICCV 2023 workshop paper "On the Adversarial Robustness of Multi-Modal Foundation Models," show that an L∞ attack of radius 1/255, run with APGD, can force OpenFlamingo (CLIP ViT-L-14 plus MPT-7B, 9B parameters in the model they emphasize) to emit a chosen caption, and that untargeted attacks damage caption and VQA scores. The threat they emphasize is a malicious image shown to an honest user. The prompt is not a factorial context variable.

Taken together, these three papers establish that pixels can control a vision-language model's text, and that the words beside the image change how much a given pixel attack matters. They leave the leakage-controlled, same-question contrast unmeasured. Study 1, reviewed in §7, is a small version of that contrast in which the supportive words contained the answer.

## 3. Gradient alignment and transfer

### 3.1 Demontis et al.

Demontis, Melis, Pintor, Jagielski, Biggio, Oprea, Nita-Rotaru, and Roli, USENIX Security 2019, analyze evasion and poisoning transfer from a surrogate classifier to a target classifier. Under a linearization of the loss, the relative increase from an L2 step is the cosine of the two input gradients:

\[
R(x, y) = \frac{\nabla_x \hat{\ell}^\top \nabla_x \ell}{\lVert \nabla_x \hat{\ell} \rVert_2 \, \lVert \nabla_x \ell \rVert_2}.
\]

They also define the size of the target gradient and the variance of the surrogate loss across training sets. On digit, malware, and face classifiers, gradient alignment tracks transfer. The surrogate and the target are different models, or the same algorithm trained on different samples. They are not two text contexts of one network. The paper says that other perturbation norms change the alignment formula. That sentence is why Study 2's \(A(s, t)\) is written for an L∞ sign step rather than copied as cosine similarity. The proposal stores cosine as a diagnostic and does not pretend the two formulas are the same.

### 3.2 Madry et al. and Goodfellow et al.

Madry, Makelov, Schmidt, Tsipras, and Vladu (`arxiv.org/pdf/1706.06083v4`) define projected gradient descent inside an L∞ ball,

\[
x^{t+1} = \Pi_{x+\mathcal{S}}\big(x^t + \alpha \operatorname{sign}(\nabla_x L(\theta, x, y))\big),
\]

and argue from experiments that it behaves as a universal first-order adversary. The same PDF reports adversarial perturbations that have a negative inner product with the loss gradient, and a correlation with the gradient direction that gets worse as the perturbation grows. That is direct evidence that a first-order alignment need not predict a multi-step PGD outcome. The proposal therefore treats FGSM as the linear control and treats a PGD-only association as a stronger, less obvious claim.

Goodfellow, Shlens, and Szegedy (`arxiv.org/pdf/1412.6572`) introduce the fast gradient sign method,

\[
\eta = \varepsilon \operatorname{sign}\big(\nabla_x J(\theta, x, y)\big),
\]

from a linear approximation of the cost. FGSM in Study 2 is this one-step method applied to \(L_c\), then clipped to the image range. It is a control, not a contribution.

### 3.3 A 2026 preprint that measures cross-model cosine on retrieval models

Yu, Cao, Qiang, Chen, Huang, and Liu (`arxiv.org/html/2606.10571`) define DeBias-Attack. The opened page is a preprint: the header still carries a placeholder publication identifier. They measure the cosine of input gradients between a CLIP surrogate and other vision-language pretraining models, and they remove from the attack gradient the component aligned with a gradient computed on a weak-semantic image. The alignment they plot is cross-model, on retrieval losses, not cross-context on a generative answer loss. The page calls the method the first transfer attack that corrects surrogate-specific bias by that gradient correction. That priority claim is theirs. The useful fact for Study 2 is narrower and directly visible in the paper: input-gradient cosine is already being used as a diagnostic of cross-model transfer on multimodal embedding models.

### 3.4 Established versus open

Established: for distinct classifiers, gradient cosine predicts linearized L2 transfer (Demontis et al.); multi-step L∞ attacks can depart from the gradient direction (Madry et al.); one signed step is FGSM (Goodfellow et al.); at least one 2026 preprint measures cross-model input-gradient cosine on vision-language retrieval.

Open in the opened set: the scene-level regression of strict error under context \(t\) on \(A(s, t) = \langle g_t, \operatorname{sign}(g_s) \rangle / \lVert g_t \rVert_1\), for one generative vision-language model, with \(s\) and \(t\) free of the answer string.

## 4. Multimodal gradient conflicts

The phrase "gradient conflict" already has technical meanings. They are about training, and they are not the quantity \(A(s, t)\).

Yu, Kumar, Gupta, Levine, Hausman, and Finn, NeurIPS 2020, define two task gradients as conflicting when their cosine is negative. PCGrad projects each parameter-gradient onto the normal plane of the other. The gradients are derivatives of task losses with respect to shared weights during multi-task training.

Huang, Lin, Zhou, Yang, and Huang (`arxiv.org/html/2203.12221`) prove a modality-competition result for a late-fusion network with smoothed ReLU encoders, trained by gradient descent on a simplified distribution: only a subset of modality encoders learn the features, and that failure is the source of a train-both-modalities gap. They attribute the empirical gap they set out to explain to Wang et al. 2020, which was not opened. The theorem's scope is the architecture and the data model in the paper, not a frozen vision-language model at attack time.

Peng, Wei, Deng, Wang, and Hu, CVPR 2022, show audio-visual models in which the stronger modality contributes a lower joint loss and then sends little gradient into the weaker modality. On-the-fly gradient modulation rescales training gradients. The conflict is between modality encoders during learning.

Wei, Luo, and Luo, ICCV 2025, argue that the fusion module itself reduces the gradient that reaches each encoder, including the encoder of the stronger modality, and they detach encoder updates from the fusion loss. Again the setting is training.

A negative \(A(s, t)\) has a family resemblance to these definitions: two derivatives point against each other. The derivatives in Study 2 are image gradients of one frozen model under two prompts, evaluated at test time. Applying PCGrad, gradient modulation, or disentangled gradient learning to that pair would be a different project. H\* only measures the conflict and asks whether it ranks transfer. It does not propose a training fix.

The anonymous GrCPA manuscript's "non-stationarity" is closer in time: it is instability while an image attack is optimized on many prompts at once. That is a statement about the attack trajectory. \(A(s, t)\) is a property of the clean image, before the trajectory. Both can be true. A clean-image alignment can be high while a long multi-prompt optimization still wanders, and the reverse.

## 5. Counterfactual visual grounding

Four different uses of "counterfactual" showed up in the sources. They should not be collapsed.

**Scene diagnostics with exact answers.** Johnson, Fei-Fei, Hariharan, Zitnick, van der Maaten, and Girshick (`arxiv.org/abs/1612.06890`) introduce CLEVR: rendered 3D shapes, questions generated as functional programs, and an explicit fight against language shortcuts. The questions cover attributes, counts, comparisons, and relations. The paper says that accuracy on CLEVR is not an end in itself, because a hand-built CLEVR solver need not transfer to photographs. The generator license opened on GitHub is BSD for the software. That license check is recorded in the proposal. CLEVR is a candidate later external set, not the Study 2 training distribution, and not a source of the four context keys.

**Language bias as a causal direct effect.** Niu, Tang, Zhang, Lu, Hua, and Wen, in the CVPR 2021 PDF "Counterfactual VQA: A Cause-Effect Look at Language Bias" (also `arxiv.org/abs/2006.04315`), write language bias as the direct effect of the question on the answer and subtract it from the total effect. They distinguish "bad" bias (bananas are yellow) from "good" context (a color question narrows the answer type). Experiments use VQA v2 and the language-bias-sensitive VQA-CP split. The intervention is in the model's inference graph. The image is not edited into a matched sibling with a new ground-truth answer.

**Linguistic presuppositions.** Zhang, Zhai, Zhao, Zong, Wen, and Zhao (`arxiv.org/html/2310.06627v3`) build C-VQA by editing questions with presuppositions such as "if the TV was off," and report large drops, including drops for strong multimodal models. Li, Tian, Jiao, Chen, and Jiang (`arxiv.org/html/2404.12966v2`) build CFMM and report that models answer from what is visible and ignore the counterfactual clause in the question. In both benchmarks the image stays, and the counterfactual lives in the sentence. Study 2's `contradictory` context is a relative of this design. Study 1 already showed that a false caption can destroy clean accuracy, which is why `contradictory` is not the primary pair.

**Visual patterns that embedding models miss.** Tong, Liu, Zhai, Ma, LeCun, and Xie (`arxiv.org/pdf/2401.06209`) construct MMVP from CLIP-blind image pairs: images the contrastive embedding treats as similar and people do not. They report that many multimodal LLMs, on straightforward questions about nine visual patterns, perform below chance, with GPT-4V as the exception in that summary. The lesson for Study 2 is that a correct-looking answer rate can still hide a visual failure. It is not a license to skip a clean-accuracy gate. A model that fails MMVP-style pairs can still be attacked; the attack is then hard to interpret as a change in a visual decision the model was making.

**What Study 2 means by a matched counterfactual.** Inside a group, the image changes so that \(y\) changes, and the `neutral` and `helpful_nonrevealing` strings do not change. The question can stay fixed. This is in the spirit of CLEVR's exact labels and of Niu et al.'s worry about language shortcuts. It is not Niu et al.'s subtraction, not a C-VQA presupposition, and not an MMVP pair. The proposal states that design as a requirement on the generator. It does not claim the generator has been built.

## 6. Causal interventions in vision-language models

Palit, Pandey, Arora, and Liang (`arxiv.org/pdf/2308.14179`) publish a causal-tracing tool for BLIP. They corrupt image embeddings, then patch clean hidden states into the corrupted run, and score the restored probability of the VQA answer. On the COCO-QA slice they analyze, causal effect concentrates in later layers. Clean accuracy on that slice is 80.23% for color, 26.30% for location, and 3.27% for object counting. Tracing a behavior the model does not have is not informative. Any exploratory tracing in Study 2 should start from a slice with a high clean accuracy, which is the same gate H\* already uses.

Li, Qu, Zhou, Qin, Yang, and Zhao (`arxiv.org/html/2503.06169v2`) draw a causal graph for hallucination in which vision, text, or their interaction can influence the output while bypassing fusion. They estimate natural direct effects with masked images, hallucinated text embeddings, and degraded visual inputs, then apply a test-time intervention. The opened page presents this as a hallucination mitigation method. It does not estimate attack transfer.

Neither paper is an argument for putting internal interventions into the primary confirmatory claim. Both are arguments that the exploratory track needs a pre-specified behavior the model actually exhibits, and a pre-specified intervention, and that those choices are easy to overclaim.

## 7. Study 1, read as prior art for this repository

Sources opened in `fcavaletto/crossmodal-vlm-robustness` at commit `e73326b28deae5fe50dacd078eedd0c9c960aec3`: `README.md`, `docs/PROTOCOL.md`, `docs/freeze-config.md` (through the model and data pins), `docs/CONSOLIDATION.md`, `docs/pre-test-addendum-2026-11-05.md`, and the abstract, related-work table, protocol summary, and result paragraphs of `reports/technical_report.md`.

The frozen question was whether misleading text amplifies additional PGD error relative to supportive text. The primary estimand was an image-level difference-in-differences. The model was SmolVLM2-500M-Video-Instruct at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, float32, eager attention, Apple MPS. Attacks used mean answer-token negative log likelihood, L∞ radii \(\{2, 4, 8\}/255\), PGD for 7 steps at step size \(\varepsilon/4\) with no random start, FGSM, and uniform L∞ noise. Crafting used the neutral context only. The same pixels were evaluated under neutral, supportive, and misleading text. Supportive text contained the true label. Misleading text contained a wrong label.

Held-out color PGD at ε = 8/255 gave a primary difference-in-differences of −0.500 with a 95% bootstrap interval [−0.688, −0.344] on 32 images. The misleading-color clean floor was 6/64 on development and 4/32 on the test split. A secondary neutral-minus-supportive contrast was positive for color. Held-out color-neutral conditional PGD success was 100% at ε = 2/255 and 93.1% at ε = 8/255. The authors refuse to rewrite the primary claim to match the secondary contrast. They list answer copying, the clean floor, attack saturation, and "optimize under neutral only" as limitations. The technical report's related-work table already says that context changing robustness is not new, citing Cui et al., which this review opened separately.

Consequences for Study 2, each of them a design constraint rather than a new empirical claim:

- A context that contains \(y\) is a copying control.
- A context that asserts a false label can remove the clean-correct population an attack contrast needs.
- Crafting under one context and evaluating under another is the definition of cross-context transfer, and Study 1 only did it from neutral.
- A seven-step attack can saturate a tiny synthetic task at ε = 2/255. H\* needs a budget rule that detects saturation instead of interpreting a flat error rate.
- Mean versus sum token loss does not change PGD signs and does change the numeric loss. The proposal records both.

## 8. What this proposal might add

The possible addition is one pre-registered association: β_A in the proposal, on `neutral` and `helpful_nonrevealing`, with group-invariant text, a noise control, an FGSM control, and a non-identifiable outcome if the attack saturates. Secondary contrasts restate published comparisons (context wording; Multi-P versus Single-P) on a scene family those papers did not use. That is a change of test bed, and the proposal says so.

The proposal does not add a new multi-context optimizer, a new causal-tracing algorithm, a new theory of modality competition, or a claim that vision-language models are robust or unsafe in deployment.

## 9. Sources opened

Paths are the pages or files read on 6 November 2026.

1. Luo, Gu, Liu, Torr. An Image Is Worth 1000 Lies. arXiv HTML `https://arxiv.org/html/2403.09766` and ICLR 2024 proceedings PDF `https://proceedings.iclr.cc/paper_files/paper/2024/file/674ad201bc8fa74b3c9979230aa0c63b-Paper-Conference.pdf`.
2. Anonymous. Gradient Regularization-based Cross-Prompt Attacks on Vision Language Models. OpenReview PDF `https://openreview.net/pdf?id=I05Z6KjQ9K`. Banner: under review, ICLR 2025.
3. Mittal, Pandey, Tiwari, Jindal, Swain. Revisiting CroPA. `https://arxiv.org/html/2506.22982v1`.
4. Cai, Liu, Qu, Fang, Dong, Tang, Zhou, Sun, Hu. Towards Building Model/Prompt-Transferable Attackers against Large Vision-Language Models. NeurIPS 2025 PDF `https://proceedings.neurips.cc/paper_files/paper/2025/file/fe5e73e8817e3b4d4d0a2b9a7396495f-Paper-Conference.pdf`.
5. Zhao, Pang, Du, Yang, Li, Cheung, Lin. On Evaluating Adversarial Robustness of Large Vision-Language Models. NeurIPS 2023 PDF `https://proceedings.neurips.cc/paper_files/paper/2023/file/a97b58c4f7551053b0512f92244b0810-Paper-Conference.pdf`.
6. Schlarmann, Hein. On the Adversarial Robustness of Multi-Modal Foundation Models. ICCV 2023 workshop PDF `https://openaccess.thecvf.com/content/ICCV2023W/AROW/papers/Schlarmann_On_the_Adversarial_Robustness_of_Multi-Modal_Foundation_Models_ICCVW_2023_paper.pdf`.
7. Cui, Aparcedo, Jang, Lim. On the Robustness of Large Multimodal Models Against Image Adversarial Attacks. CVPR 2024 PDF `https://par.nsf.gov/servlets/purl/10539722` and arXiv abstract `https://arxiv.org/abs/2312.03777`.
8. Demontis, Melis, Pintor, Jagielski, Biggio, Oprea, Nita-Rotaru, Roli. Why Do Adversarial Attacks Transfer? USENIX Security 2019 PDF `https://www.usenix.org/system/files/sec19-demontis.pdf`.
9. Madry, Makelov, Schmidt, Tsipras, Vladu. Towards Deep Learning Models Resistant to Adversarial Attacks. `https://arxiv.org/pdf/1706.06083v4.pdf`.
10. Goodfellow, Shlens, Szegedy. Explaining and Harnessing Adversarial Examples. `https://arxiv.org/pdf/1412.6572.pdf`.
11. Yu, Cao, Qiang, Chen, Huang, Liu. Improving Adversarial Transferability on Vision-Language Pre-training Models via Surrogate-Specific Bias Correction. `https://arxiv.org/html/2606.10571`.
12. Yu, Kumar, Gupta, Levine, Hausman, Finn. Gradient Surgery for Multi-Task Learning. NeurIPS 2020. `https://arxiv.org/pdf/2001.06782`.
13. Huang, Lin, Zhou, Yang, Huang. Modality Competition. `https://arxiv.org/html/2203.12221`.
14. Peng, Wei, Deng, Wang, Hu. Balanced Multimodal Learning via On-the-Fly Gradient Modulation. CVPR 2022 PDF `https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf`.
15. Wei, Luo, Luo. Boosting Multimodal Learning via Disentangled Gradient Learning. ICCV 2025 PDF `https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_Boosting_Multimodal_Learning_via_Disentangled_Gradient_Learning_ICCV_2025_paper.pdf`.
16. Johnson, Fei-Fei, Hariharan, Zitnick, van der Maaten, Girshick. CLEVR. `https://arxiv.org/abs/1612.06890`.
17. CLEVR generator license. `https://raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/main/LICENSE`. BSD software license, copyright 2017 Facebook.
18. Niu, Tang, Zhang, Lu, Hua, Wen. Counterfactual VQA. CVPR 2021 PDF `https://openaccess.thecvf.com/content/CVPR2021/papers/Niu_Counterfactual_VQA_A_Cause-Effect_Look_at_Language_Bias_CVPR_2021_paper.pdf` and `https://arxiv.org/abs/2006.04315`.
19. Zhang, Zhai, Zhao, Zong, Wen, Zhao. What If the TV Was Off? (C-VQA). `https://arxiv.org/html/2310.06627v3`.
20. Li, Tian, Jiao, Chen, Jiang. Eyes Can Deceive (CFMM). `https://arxiv.org/html/2404.12966v2`.
21. Tong, Liu, Zhai, Ma, LeCun, Xie. Eyes Wide Shut? (MMVP). `https://arxiv.org/pdf/2401.06209`.
22. Palit, Pandey, Arora, Liang. Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP. `https://arxiv.org/pdf/2308.14179`.
23. Li, Qu, Zhou, Qin, Yang, Zhao. Treble Counterfactual VLMs. `https://arxiv.org/html/2503.06169v2`.
24. Hugging Face model card, SmolVLM2-500M-Video-Instruct. `https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
25. Hugging Face model card, Qwen3-VL-2B-Instruct. `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct`.
26. Cavaletto. crossmodal-vlm-robustness, commit `e73326b28deae5fe50dacd078eedd0c9c960aec3`. Files listed in §7. `https://github.com/fcavaletto/crossmodal-vlm-robustness`.
