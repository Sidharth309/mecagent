from datasets import load_dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load Dataset
dataset = load_dataset("./GenCAD-Code", num_proc=16, split=["train", "test"])
train_ds, test_ds = dataset
print(train_ds[0].keys())

# Load BLIP-2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# Generate
prompt = "Generate CadQuery code for the object in the image."
inputs = processor(images=train_ds[0]["image"], text=prompt, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=300)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(output)


from mecagent_technical_test.metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from mecagent_technical_test.metrics.best_iou import get_iou_best

gt_code = train_ds[0]["cadquery"]
pred_code = output

vsr = evaluate_syntax_rate_simple({"pred": pred_code})
iou = get_iou_best(gt_code, pred_code)

print(f"Baseline VSR: {vsr:.3f}, IOU: {iou:.3f}")






# Enhancement: LoRA Fine-Tuning on BLIP-2

#Step 1: Define the LoRA Config and Wrap the Model
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load the processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16).cuda()

# Apply LoRA config to the model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Code is treated as text output
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q", "v"]  # You may need to inspect the model for correct target modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Shows only LoRA layers will be trained

#Step 2: Prepare Dataset
from datasets import load_dataset

ds = load_dataset("./GenCAD-Code", num_proc=16)
train_ds, test_ds = ds["train"], ds["test"]

def preprocess(example):
    prompt = "Generate CadQuery code for the object in the image."
    inputs = processor(images=example["image"], text=prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = processor.tokenizer(example["cadquery"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return {k: v.squeeze(0) for k, v in inputs.items()}

processed_train = train_ds.map(preprocess, batched=False)


#Step 3: Fine-tune the Model with LoRA
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./blip2-lora-cadquery",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
)
trainer.train()

#Step 4: Evaluate
from mecagent_technical_test.metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from mecagent_technical_test.metrics.best_iou import get_iou_best

example = test_ds[0]
inputs = processor(images=example["image"], text="Generate CadQuery code for the object in the image.", return_tensors="pt").to("cuda")
output_ids = model.generate(**inputs, max_new_tokens=300)
output_code = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

vsr = evaluate_syntax_rate_simple({"pred": output_code})
iou = get_iou_best(example["cadquery"], output_code)

print(f"LoRA Fine-Tuned VSR: {vsr:.3f}, IOU: {iou:.3f}")






# Enhancement: Multiview learning for CadQuery code generation

#Step 1: Multi-View Data Augmentation

#Generate rotated renderings (yaw/pitch)
import cadquery as cq
from cadquery import exporters
import tempfile
from pathlib import Path
import trimesh
import pyrender
import numpy as np
from PIL import Image

def render_views(solid, angles=[(0, 0), (30, 0), (0, 30), (30, 30)]):
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = Path(tmpdir) / "tmp.stl"
        exporters.export(solid, str(stl_path))
        mesh = trimesh.load_mesh(str(stl_path))

        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)

        images = []
        for yaw, pitch in angles:
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            cam_pose = trimesh.transformations.euler_matrix(
                np.radians(pitch), np.radians(yaw), 0
            )
            scene.add(camera, pose=cam_pose)

            r = pyrender.OffscreenRenderer(224, 224)
            color, _ = r.render(scene)
            images.append(Image.fromarray(color))
            scene.remove_node(scene.get_nodes(name='camera')[-1])  # remove camera
        return images


#Overlay annotations (axes, labels, etc)
from PIL import ImageDraw

def draw_axes(image):
    draw = ImageDraw.Draw(image)
    draw.line((112, 112, 180, 112), fill="red", width=3)  # X-axis
    draw.line((112, 112, 112, 40), fill="green", width=3)  # Y-axis
    draw.line((112, 112, 140, 140), fill="blue", width=3)  # Z-axis
    return image


#Step 2: Model Architecture for Multi-View BLIP-2

# Concatenate views horizontally
from PIL import Image

def concatenate_views(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


#Step 3: Training
from mecagent_technical_test.metrics.valid_syntax_rate import load_solid_from_code

def preprocess_multi_view(example):
    prompt = "Generate CadQuery code from multiple views of the object."
    views = render_views(load_solid_from_code(example["cadquery"]))  # or cached
    img = concatenate_views([draw_axes(v) for v in views])
    inputs = processor(images=img, text=prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = processor.tokenizer(example["cadquery"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return {k: v.squeeze(0) for k, v in inputs.items()}


#Step 4: Evaluation
evaluate_syntax_rate_simple({"id": pred_code})
get_iou_best(gt_code, pred_code)
