import qai_hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def run_inference(model, device, input_dataset):
    """Submits an inference job for the model and waits for completion."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1"
    )
    inference_job.wait()
    return inference_job.job_id


def parse_ground_truth(txt_list, img_list):
    df_img = pd.read_csv(img_list)
    df_txt = pd.read_csv(txt_list)
    txt_id = df_txt.iloc[:, 0].dropna().astype(np.int16).tolist()
    gt = df_img.iloc[:, 1].dropna().tolist()
    return txt_id, gt


def evaluate_track1(img_output, txt_output, txt_list, img_list, k=10):
    """
    Compute Recall@K between image and text embeddings.

    Args:
        img_output (list): Image encoder outputs, each shape (1, D)
        txt_output (list): Text encoder outputs, each shape (1, D)
        txt_list (str): Path to text CSV
        img_list (str): Path to image CSV
        k (int): Top-K for recall computation

    Returns:
        float: Mean Recall@K
    """
    img_embeds = np.vstack([x for x in img_output])   # [N, D]
    txt_embeds = np.vstack([x for x in txt_output])   # [M, D]

    # L2-normalize before cosine similarity
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)

    sim_matrix = cosine_similarity(img_embeds, txt_embeds)   # [N, M]

    txt_id, gt = parse_ground_truth(txt_list, img_list)

    recalls = []
    for i in range(len(img_embeds)):
        gt_ids = [int(x) for x in gt[i].split(';')]
        top_k_indices = np.argsort(-sim_matrix[i])[:k]
        predicted_txt_ids = [txt_id[idx] for idx in top_k_indices]
        matched = len(set(predicted_txt_ids) & set(gt_ids))
        recalls.append(matched / len(gt_ids))

    return np.mean(recalls)


# --- Define target device ---
device = qai_hub.Device("XR2 Gen 2 (Proxy)")

# TODO: Fill in compiled job IDs and dataset IDs after running compile_and_profile.py + upload_dataset.py
tasks = {
    "text": {
        "compiled_id": "j5w9yq33p",    # text encoder compile job ID
        "dataset_id":  "d9edokom2"     # text dataset ID from upload_dataset.py
    },
    "image": {
        "compiled_id": "jpev6kwo5",    # image encoder compile job ID
        "dataset_id":  "d7jzqoqz2"     # image dataset ID from upload_dataset.py
    }
}

outputs = {}
for task_name, info in tasks.items():
    compiled_id  = info["compiled_id"]
    input_dataset = qai_hub.get_dataset(info["dataset_id"])

    job = qai_hub.get_job(compiled_id)
    compiled_model = job.get_target_model()

    print(f"Running inference for {task_name} ({compiled_model.model_id}) on {device.name}...")
    inference_id  = run_inference(compiled_model, device, input_dataset)
    inference_job = qai_hub.get_job(inference_id)

    if inference_job.get_status().failure:
        print(f"{task_name.capitalize()} inference failed")
        outputs[task_name] = None
    else:
        inference_output = inference_job.download_output_data()
        outputs[task_name] = inference_output["output_0"]

result = evaluate_track1(
    outputs["image"], outputs["text"],
    txt_list="dataset/txt_list.csv",
    img_list="dataset/img_list.csv"
)
print(f"\nRecall@10: {result:.4f}")
