# Face DeStyle Pipeline：项目展示叙事

## 一句话版本

围绕“风格化人像还原为自然照片”搭建了一套多风格生成与评估管线，并通过视觉实验发现不同媒介需要不同处理深度，最终形成 Base 单阶段、顺序二阶段、语义重建和有限 LoRA 适配组成的人工审核路由。

## 项目背景

风格化人像的困难不只是去掉颜色或纹理。漫画和水彩主要改变二维外观；3D、Clay、针毡和折纸还会改变眼睛、头部、头发、服装、底座等几何与材质语义。因此，给所有风格套用同一个提示词或同一个适配器，容易出现两类问题：风格残留，或为了追求真人感而改变人物年龄、表情、姿态和场景。

本项目借鉴 *Learning to Stylize by Learning to Destylize* 的研究方向，但实现的是独立的小规模复现与工程探索。目标不是声称恢复人物的真实外貌，而是在保留可见人物与场景证据的前提下，研究不同风格的去风格化处理边界。

## 方法流程

1. 建立 Comic、Ink、Watercolor、3D cartoon、Clay、Needle-felt 和 Origami 七类风格数据与来源记录。
2. 使用 FLUX.1-Kontext-dev 进行原生图像指令编辑，以风格自适应提示词替代单一通用描述。
3. 对简单二维媒介优先采用一次 Base 编辑；仅在残余材质明显时运行真顺序 Stage 2。
4. 对改变人物构造的针毡胸像采用语义重建提示，将底座和毛毡结构还原为合理的肩颈、躯干与衣物。
5. 对 Origami 建立严格筛选的配对数据并训练 LoRA，通过固定留出样本比较 Base 与多个 checkpoint。
6. 以风格去除、人物证据、姿态构图和场景保持进行人工视觉审核，输出实际处理路由而不是强行选出一个全局模型。

## 最值得展示的结果

- 137 张非 Origami 源图完成两阶段推理，共得到 274 张输出；人工审核选择 71 张可用结果。
- Comic、Ink、Watercolor 分别有 23/24、21/24、21/24 的 Stage 1 可用结果，说明二维外观型风格通常不需要二次编辑。
- Clay 的 Stage 2 从 Stage 1 严格 0/24 中救回 5 个样本，形成了清晰的“残余材质再处理”案例。
- Origami V1 checkpoint 100 在六张固定留出图上约 3/6 通过，优于 Base 约 1/6；继续增加困难配对和修改 caption 没有稳定提升，因此保留早期 checkpoint，停止过度训练。
- 后续展示扩展为 3D 和 Needle-felt 各运行 10 张两阶段候选：3D 得到 1 张主展示与 3 张备选，针毡得到 4 张稳定的语义重建展示图。该扩展用于案例展示，不并入 137 张正式统计。

## 这项工作的核心思路

项目的重点不是“多跑模型”，而是把失败模式转化为路由设计：

- 二维纹理残留较少：一次编辑后直接输出。
- Clay 等局部材质仍明显：将 Stage 1 输出送入第二次残余编辑。
- 针毡胸像存在底座和非人体结构：切换到人物语义重建，而不是机械保留每一个轮廓。
- 3D 几何和渲染感难以同时消除：保留人工审核和教师模型作为受控后备，不宣称通用成功。
- Origami 适配器在部分样本有效：选择较早 checkpoint，避免后期材质去除增强带来的年龄、表情和身份线索漂移。

这使项目从“一套提示词处理所有风格”演进为“根据媒介失真类型选择处理深度”的可解释系统。

## 简历表述（推荐）

**Face DeStyle Pipeline｜多风格人像去风格化与处理路由**

- 基于 FLUX.1-Kontext-dev 构建覆盖 7 类视觉风格的图像编辑管线，完成数据清单、风格自适应提示、两阶段顺序推理、LoRA 训练和人工视觉审核的端到端实验流程。
- 在 137 张非折纸验证图上完成 274 次两阶段生成并筛选 71 张可用结果；据失败模式设计 Base 单阶段、Clay 二阶段救援、困难材质语义重建和显式失败处理路由。
- 为 Origami 构建 23 对严格筛选训练样本并训练 rank-16 LoRA；通过固定六图留出比较选择 checkpoint 100，将严格通过从 Base 约 1/6 提升至约 3/6，同时识别后期 checkpoint 的人物漂移问题。

若简历空间只允许一条：

> 基于 FLUX.1-Kontext-dev 搭建七风格人像去风格化管线，在 137 张验证图上完成 274 次顺序生成与人工审核，并据风格残留、结构漂移和人物保持设计单阶段、二阶段、语义重建及有限 LoRA 适配路由。

## English resume version

**Face DeStyle Pipeline — Multi-style portrait destylization and routing**

- Built an end-to-end FLUX.1-Kontext-dev image-editing pipeline across seven visual styles, covering provenance-aware data manifests, style-adaptive prompts, sequential refinement, LoRA adaptation, and visual review.
- Generated 274 two-stage outputs for 137 non-origami sources and selected 71 usable reconstructions; converted observed failure modes into a review-gated router spanning direct Stage 1, Clay residual refinement, semantic reconstruction, and explicit failure handling.
- Curated 23 Origami training pairs and trained a rank-16 LoRA; fixed-holdout comparison selected checkpoint 100, improving strict qualitative passes from about 1/6 with Base to about 3/6 while exposing late-checkpoint subject drift.

## 面试时的 90 秒讲法

“这个项目研究的是风格化人像如何还原为自然照片。我最初尝试统一提示和结构控制，但发现漫画、水彩这类二维风格与 3D、Clay、折纸这类几何材质风格的失败模式完全不同。所以我把任务改造成一个处理路由：Comic、Ink 和 Watercolor 通常一次 FLUX 编辑就够；Clay 在第一次编辑后仍有材质残留，因此只对必要样本做顺序二次精修；针毡胸像需要把底座和毛毡构造重建成肩颈和衣物；Origami 则使用严格筛选的配对数据训练 LoRA，并通过留出图选择较早 checkpoint，避免继续训练造成年龄和表情漂移。最后我在 137 张非折纸图上完成 274 次生成并人工审核，得到 71 张可用结果。项目最有价值的地方不是某一个最好看的样本，而是从失败分析中形成了一套可解释的多风格处理策略。”

## 展示顺序

1. 先放七风格总览，证明系统覆盖面。
2. 用 Comic / Ink / Watercolor 对照说明简单风格的一阶段直达。
3. 用 Clay 三列图说明为什么需要真顺序 Stage 2。
4. 用 Needle-felt 三列图解释严格轮廓保持与语义重建的区别。
5. 用 3D 渐进图展示开放模型的边界以及教师结果提供的上限信号。
6. 最后用 Origami Base / V1-100 对照讲数据筛选、LoRA 和 checkpoint 选择。

本地成图位于：

```text
/Users/pot/Documents/大创/实验归档/portfolio-showcase-20260827
```

## 表述边界

- 可以说“从风格化输入重建自然人像”，不要说“恢复人物真实长相”或“真实身份”。
- 可以突出代表性成功案例，但 71/137 的最终审核结果和各路线样本量需要在技术说明中保留。
- 3D 和 Needle-felt 的 20 图补充是展示导向扩展，不应并入正式 137 图统计。
- V1 checkpoint 100 是有限适配器，不应写成普适 Origami 解决方案。
