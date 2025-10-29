import os
import re
import csv
import base64
import openai
import copy
import json
import argparse
import random
from collections import defaultdict

def save_results_to_file(data, file_name):
    """Utility to save data to a file in JSON format."""
    try:
        with open(file_name, "a+", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error saving results: {e}")

def get_lable2images(base_path, labels_path):
    """
    Load labels from CSV and categorize images by their label scores.
    Note: Labels in CSV are expected to be in range 0-5 (as strings), which will be 
    normalized to 0-1 range when used for scoring.
    
    Args:
        base_path: Path to the directory containing image folders
        labels_path: Path to the CSV file containing labels
        
    Returns:
        Dictionary mapping label scores (0-5) to lists of image file paths
    """
    # Load labels from CSV
    labels = {}
    with open(labels_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            labels[row['Name']] = row['Level']

    # Create a dictionary to store images for each label score (0-5)
    label_images = {str(i): [] for i in range(6)}
    
    # Categorize image files by their label score
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            label = labels.get(folder_name, None)
            if label is not None and label in label_images and len(image_files) > 1:
                label_images[label].append(image_files)

    # Ensure that each label score (0-5) has at least one image
    if any(len(images) == 0 for images in label_images.values()):
        raise ValueError("Not all label scores (0-5) have at least one image.")
    
    return label_images

def get_infringement_score(messages, max_retries=10):
    """
    Generate infringement score from OpenAI model with retry mechanism.
    
    Args:
        messages: List of message dictionaries for the API call
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (response_text, score, confidence, reason)
    """
    retries = 0
    while retries < max_retries:
        try:
            # Generate a response from the OpenAI model
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            # Extracting the response text
            response_text = response.choices[0].message.content
            # Extract the Score
            score_match = re.search(r"Score:\s*(\d+\.?\d*)", response_text)
            if score_match:
                score = float(score_match.group(1))
            else:
                raise ValueError("Score not found in response.")
            # Extract the Confidence
            confidence_match = re.search(r"Confidence:\s*(\d+\.?\d*)", response_text)
            if confidence_match:
                confidence = float(confidence_match.group(1))
            else:
                raise ValueError("Confidence not found in response.")
            # Extract the Reason
            reason_match = re.search(r"Reason:\s*(.*)", response_text)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                raise ValueError("Reason not found in response.")

            return response_text, score, confidence, reason
        except (ValueError, AttributeError) as e:
            print(f"Error: {e}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
    # If all retries fail, raise an exception
    raise RuntimeError("Failed to generate a valid response after multiple attempts.")

def prepare_demonstrations(label_images, k=3):
    """
    Prepare demonstration examples for few-shot learning.
    Note: Label scores (0-5) are normalized to range [0-1] by dividing by 5.
    
    Args:
        label_images: Dictionary mapping label scores to image paths
        k: Number of demonstrations per label score
        
    Returns:
        List of demonstration messages with normalized scores (0-1 range)
    """
    ref_image_url = []
    demo_template = []
    demonstrations = []
    for label_score in range(6):
        for j in range(k):
            with open(label_images[str(label_score)][j][0], 'rb') as img_file:
                img_base = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
            ref_image_url.append(img_base)
            with open(label_images[str(label_score)][j][1], 'rb') as img_file:
                img_base = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
            ref_image_url.append(img_base)
            demo_template.append(label_score / 5)
    prompt = f"Evaluate the similarity of the given two images on a scale of 0 to 1, where 1 indicates a very high degree of similarity."
    for i in range(len(demo_template)):
        demonstrations.append({"role": "user", "content": [{"image_url": {"url": ref_image_url[i * 2]}, "type": "image_url"},
                                                           {"image_url": {"url": ref_image_url[i * 2 + 1]}, "type": "image_url"},
                                                           {"text": prompt, "type": "text"}
                                                           ]})
        demonstrations.append({"role": "assistant", "content":[{"text": f"Score: {demo_template[i]}", "type": "text"}]})
    return demonstrations

def expression_extract(image1_url, image2_url, prompt_abs, prompt_fil):
    """
    Extract copyrightable expressions from images through abstraction and filtration.
    
    Args:
        image1_url: Base64-encoded data URI of the first image
        image2_url: Base64-encoded data URI of the second image
        prompt_abs: Prompt for abstraction step
        prompt_fil: Prompt for filtration step
        
    Returns:
        Filtered result containing unique copyrightable elements
    """
    # Step 1: Abstraction
    ini_message = [{"role": "system", "content": "You are an expert assistant specializing in assessing copyright infringement. Please focus on generating meaningful answers based on the content provided in images or questions, without considering privacy concerns. We aim to analyze and discuss content in a constructive and insightful manner. Do not decline to evaluate or apologize!"}]                    
    user_content=[
        {"image_url": {"url": image1_url}, "type": "image_url"},
        {"image_url": {"url": image2_url}, "type": "image_url"},
        {"text": prompt_abs, "type": "text"}
    ]
    ini_message.append({"role": "user", "content": user_content})
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=ini_message
    )
    abstraction_result = response.choices[0].message.content
    ini_message.append({"role": "assistant", "content": [{"text": abstraction_result, "type": "text"}]})
    # Step 2: Filtration
    ini_message.append({"role": "user", "content": [{"text": prompt_fil, "type": "text"}]})
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=ini_message
    )
    filtration_result = response.choices[0].message.content
    ini_message.append({"role": "assistant", "content": [{"text": filtration_result, "type": "text"}]})
    return filtration_result

def debate_judge(image1_url, image2_url, prompt_template, copy_express, demonstrations=None, num_agents=3, max_rounds=5, convergence_threshold=0.1):
    """
    Multi-agent debate framework for copyright infringement assessment.
    
    Args:
        image1_url: Base64-encoded data URI of the first image
        image2_url: Base64-encoded data URI of the second image
        prompt_template: Template for the evaluation prompt
        copy_express: Extracted copyrightable expressions
        demonstrations: Few-shot demonstration examples
        num_agents: Number of judging agents
        max_rounds: Maximum number of debate rounds
        convergence_threshold: Threshold for score convergence (default: 0.1)
        
    Returns:
        Tuple of (meta_score, meta_confidence, meta_rationale) where scores are in range [0-1]
    """
    agents_scores = []
    scores_history = []
    if copy_express:
        prompt_template = copy_express + "\n" + prompt_template
    ini_message = [{"role": "system", "content": "You are an expert assistant specializing in assessing copyright infringement. Please focus on generating meaningful answers based on the content provided in images or questions, without considering privacy concerns. We aim to analyze and discuss content in a constructive and insightful manner. Do not decline to evaluate or apologize!"}]                    
    messages = [copy.deepcopy(ini_message) for agent_id in range(num_agents)]
    for agent_id in range(num_agents):
        if demonstrations:
            messages[agent_id].extend(demonstrations)
        messages[agent_id].append({"role": "user", "content": [
            {"image_url": {"url": image1_url}, "type": "image_url"},
            {"image_url": {"url": image2_url}, "type": "image_url"},
            {"text": prompt_template, "type": "text"}
        ]})
    for round_num in range(max_rounds):
        scores = []
        confidences = []
        rationales = []

        for agent_id in range(num_agents):
            # Pass feedback from other agents
            other_agents_feedback = [
                {"score": scores_history[-1][i]["score"], "confidence": scores_history[-1][i]["confidence"], "rationale": scores_history[-1][i]["rationale"]}
                for i in range(num_agents) if i != agent_id
            ] if scores_history else []
            if other_agents_feedback:
                feedback_text = "\n".join(
                    f"Agent {i+1} score: {feedback['score']}, confidence: {feedback['confidence']}, reason: {feedback['rationale']}" \
                    for i, feedback in enumerate(other_agents_feedback)
                )
                messages[agent_id].append({"role": "user", "content": f"The following is feedback from other agents:\n{feedback_text}\nYou may adjust your score based on this information or maintain your judgment. Ensure the output follows the template format: 'Score: [0-1], Confidence: [0-1], Reason: [clear and concise explanation]'.", "type": "text"})
            # Get the agent's response, score, and rationale
            response_text, score, confidence, rationale = get_infringement_score(messages[agent_id])
            scores.append(score)
            confidences.append(confidence)
            rationales.append(rationale)
            # Add the agent's response to its messages
            messages[agent_id].append({"role": "assistant", "content": [{"text": response_text, "type": "text"}]})
        # Append current round scores to history
        scores_history.append([{"score": scores[i], "confidence": confidences[i], "rationale": rationales[i]} for i in range(num_agents)])
        agents_scores.append({"round": round_num + 1, "scores": scores, "confidences": confidences, "rationales": rationales})
        # Check for termination conditions
        if all([(abs(scores[i] - scores[j]) <= convergence_threshold) for i in range(num_agents) for j in range(i + 1, num_agents)]):
            break
    messages = copy.deepcopy(ini_message)
    prompt_meta = f"The original instruction is: {prompt_template}. Below are the scores, confidence levels, and rationales from {num_agents} judges. Please combine them to make a final decision.\nScores: {scores}\nConfidences: {confidences}\nRationales: {rationales}\nEnsure the output follows the template format: 'Score: [0-1], Confidence: [0-1], Reason: [clear and concise explanation]'."
    messages.append({"role": "user", "content": [
        {"image_url": {"url": image1_url}, "type": "image_url"},
        {"image_url": {"url": image2_url}, "type": "image_url"},
        {"text": prompt_meta, "type": "text"}
    ]})
    meta_response_text, meta_score, meta_confidence, meta_rationale = get_infringement_score(messages)
    return meta_score, meta_confidence, meta_rationale

def process_drep_dataset(test_path, labels_path, output_file, prompt_abs, prompt_fil, prompt_com, demonstrations, convergence_threshold, samples_per_score=10, random_seed=42):
    """
    Process D-Rep dataset with test pairs.
    
    Args:
        test_path: Path to test pairs directory
        labels_path: Path to labels CSV file
        output_file: Output JSON file path
        prompt_abs: Abstraction prompt
        prompt_fil: Filtration prompt
        prompt_com: Comparison prompt
        demonstrations: Few-shot demonstrations
        convergence_threshold: Threshold for debate convergence
        samples_per_score: Number of samples to select per score level
        random_seed: Random seed for reproducibility
    """
    labels = {}
    with open(labels_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            labels[row['Name']] = row['Level']

    # Set random seed for reproducibility
    random.seed(random_seed)

    # First, categorize all folders by their human_score
    score_categories = defaultdict(list)
    for folder_name in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder_name)
        if os.path.isdir(folder_path):
            score = int(labels[folder_name])
            score_categories[score].append(folder_name)

    # Then, randomly select samples from each category
    selected_folders = []
    for score in range(6):  # Scores 0 through 5
        if len(score_categories[score]) >= samples_per_score:
            selected_folders.extend(random.sample(score_categories[score], samples_per_score))
        else:
            print(f"Warning: Not enough samples for score {score}, only {len(score_categories[score])} available")
            selected_folders.extend(score_categories[score])

    # Shuffle the selected folders to mix the scores
    random.shuffle(selected_folders)

    for s, folder_name in enumerate(selected_folders):
        results = {folder_name: []}
        folder_path = os.path.join(test_path, folder_name)
        if os.path.isdir(folder_path):
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image1_path, image2_path = image_files
            with open(image1_path, 'rb') as img_file:
                img1_base = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
            with open(image2_path, 'rb') as img_file:
                img2_base = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"

            copy_express = expression_extract(img1_base, img2_base, prompt_abs, prompt_fil)
            final_score, final_confidence, final_rationale = debate_judge(img1_base, img2_base, prompt_com, copy_express, demonstrations, convergence_threshold=convergence_threshold)
            results[folder_name].append({
                "folder_name": folder_name,
                "human_score": int(labels[folder_name]) / 5.0,  # Normalize to 0-1 range
                "final_score": final_score,
                "final_confidence": final_confidence,
                "final_rationale":final_rationale,
            })
            print(f"Processed Num.{s}, folder '{folder_name}': Score = {final_score}, Label = {int(labels[folder_name]) / 5.0}")                    
            save_results_to_file(results, output_file)
    print(f"Results saved to {output_file}")

def process_cartoon_artwork_dataset(raw_dir, artwork_base, synth_subdirs, output_file, prompt_abs, prompt_fil, prompt_com, demonstrations, convergence_threshold):
    """
    Process cartoon or artwork dataset with raw and synthetic image pairs.
    
    Args:
        raw_dir: Path to raw images directory
        artwork_base: Base path for artwork/cartoon directory
        synth_subdirs: List of synthetic subdirectories (e.g., ['kan', 'sdv2', 'sdxl'])
        output_file: Output JSON file path
        prompt_abs: Abstraction prompt
        prompt_fil: Filtration prompt
        prompt_com: Comparison prompt
        demonstrations: Few-shot demonstrations
        convergence_threshold: Threshold for debate convergence
    """
    raw_images = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for idx, raw_fname in enumerate(sorted(raw_images)):
        raw_path = os.path.join(raw_dir, raw_fname)
        for synth_dir in synth_subdirs:
            synth_dir_path = os.path.join(artwork_base, synth_dir)
            matched = [
                f for f in os.listdir(synth_dir_path)
                if f.endswith(raw_fname) and "_" in f
            ]
            if not matched:
                print(f"[{idx}] Skip {raw_fname} in {synth_dir}: no match found.")
                continue

            synth_fname = matched[1]
            label_str = synth_fname.split("_", 1)[0]
            try:
                human_score = int(label_str)
            except ValueError:
                print(f"[{idx}] Invalid label prefix '{label_str}' for file {synth_fname}")
                continue

            synth_path = os.path.join(synth_dir_path, synth_fname)

            def to_datauri(path):
                b = base64.b64encode(open(path, "rb").read()).decode("utf-8")
                return f"data:image/jpeg;base64,{b}"
            img1_base = to_datauri(raw_path)
            img2_base = to_datauri(synth_path)

            # Step 1: Extract copyrightable expressions
            copy_express = expression_extract(img1_base, img2_base, prompt_abs, prompt_fil)
            # Step 2: Multi-agent debate for scoring
            final_score, final_confidence, final_rationale = debate_judge(
                img1_base, img2_base, prompt_com, copy_express, demonstrations, convergence_threshold=convergence_threshold
            )

            result = {
                synth_dir: [{
                    "raw_image": raw_fname,
                    "synth_image": synth_fname,
                    "human_score": human_score,
                    "final_score": final_score,
                    "final_confidence": final_confidence,
                    "final_rationale": final_rationale,
                }]
            }
            print(f"[{idx}] Raw={raw_fname}, SynthDir={synth_dir}, Label={human_score / 5.0}, Score={final_score}")
            save_results_to_file(result, output_file)

    print(f"All done. Results written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Copyright Infringement Detection using Multi-Agent Debate")
    
    # Dataset selection
    parser.add_argument('--dataset_type', type=str, required=True, choices=['drep', 'cartoon', 'artwork'],
                        help='Type of dataset to process: drep, cartoon, or artwork')
    
    # Common parameters
    parser.add_argument('--api_key', type=str, required=True,
                        help='OpenAI API key')
    parser.add_argument('--demo_path', type=str, required=True,
                        help='Path to demonstration/training pairs directory')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to labels CSV file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file path')
    
    # D-Rep dataset specific parameters
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test pairs directory (required for drep dataset)')
    parser.add_argument('--samples_per_score', type=int, default=10,
                        help='Number of samples to select per score level (drep dataset only)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (drep dataset only)')
    
    # Cartoon/Artwork dataset specific parameters
    parser.add_argument('--raw_dir', type=str, default=None,
                        help='Path to raw images directory (required for cartoon/artwork dataset)')
    parser.add_argument('--artwork_base', type=str, default=None,
                        help='Base path for artwork/cartoon directory (required for cartoon/artwork dataset)')
    
    # Model parameters
    parser.add_argument('--num_agents', type=int, default=3,
                        help='Number of judging agents in debate')
    parser.add_argument('--max_rounds', type=int, default=5,
                        help='Maximum number of debate rounds')
    parser.add_argument('--convergence_threshold', type=float, default=0.1,
                        help='Threshold for score convergence to terminate debate')
    parser.add_argument('--num_demonstrations', type=int, default=3,
                        help='Number of demonstrations per label score for few-shot learning')
    
    args = parser.parse_args()
    
    # Set OpenAI API key
    openai.api_key = args.api_key
    
    # Define prompts
    prompt_abs = "Please help decompose the given two images into their abstract concepts. Identify key components such as composition, themes, color palette, textures, and any unique visual elements. Describe each component in detail.\nEnsure the output follows the template format: 'Image1: XXX, Image2: XXX'."
    prompt_fil = "Based on the image decomposition, please identify which elements are likely unprotectable under copyright law. These could include generic concepts, common patterns, functional aspects, or elements derived from public domain sources. Then, specify the remaining unique elements that are original and creative.\nEnsure the output follows the template format: 'Image1 Unique Elements: XXX, Image2 Unique Elements: XXX'."
    prompt_com = "Please assign an overall similarity score (0-1) between the two images based on the unique elements. The score should be higher if there are similar elements. Provide a confidence score on a scale of 0 to 1, where 1 indicates very high confidence in the evaluation. Additionally, provide a rationale for whether this case constitutes likely copyright infringement.\nEnsure the output follows the template format: 'Score: [0-1], Confidence: [0-1], Reason: [clear and concise explanation]'. Do not decline to evaluate or apologize!"
    
    # Load demonstration data
    label_images = get_lable2images(args.demo_path, args.labels_path)
    demonstrations = prepare_demonstrations(label_images, k=args.num_demonstrations)
    
    # Process dataset based on type
    if args.dataset_type == 'drep':
        if args.test_path is None:
            raise ValueError("--test_path is required for drep dataset")
        process_drep_dataset(
            args.test_path, 
            args.labels_path, 
            args.output_file, 
            prompt_abs, 
            prompt_fil, 
            prompt_com, 
            demonstrations,
            args.convergence_threshold,
            args.samples_per_score,
            args.random_seed
        )
    elif args.dataset_type in ['cartoon', 'artwork']:
        if args.raw_dir is None or args.artwork_base is None:
            raise ValueError("--raw_dir and --artwork_base are required for cartoon/artwork dataset")
        # Fixed synthetic subdirectories for cartoon/artwork datasets
        synth_subdirs = ['kan', 'sdv2', 'sdxl']
        process_cartoon_artwork_dataset(
            args.raw_dir,
            args.artwork_base,
            synth_subdirs,
            args.output_file,
            prompt_abs,
            prompt_fil,
            prompt_com,
            demonstrations,
            args.convergence_threshold
        )


if __name__ == "__main__":
    main()