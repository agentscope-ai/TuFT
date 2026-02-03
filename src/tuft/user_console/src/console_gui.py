import argparse
import re
from typing import List

import gradio as gr
import pandas as pd

from . import console_config
from .console_gui_helper import (
    fetch_run_detail,
    generate_api_key,
    load_ckpts,
    load_models,
    load_runs,
    run_sample,
)


def get_model_choices(model_type: str, api_key: str):
    """Return (base_models_list, ft_models_list)"""
    if not api_key:
        return [], []

    try:
        if model_type == "Base Model":
            base_models = load_models(api_key)
            ft_models = []
        elif model_type == "Fine-tuned Model":
            ckpts = load_ckpts(api_key)
            ft_models = [ckpt.get("path", "") for ckpt in ckpts if "path" in ckpt]
            base_models = []
        else:
            base_models, ft_models = [], []
        return base_models, ft_models
    except Exception as e:
        print(f"Error loading models: {e}")
        return [], []


def show_training_detail(evt: gr.SelectData, df: pd.DataFrame, api_key: str):
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row_idx < 0 or row_idx >= len(df) or "id" not in df.columns:
            raise ValueError("Invalid row selection")

        run_id = str(df.iloc[row_idx]["id"]).strip()
        if not run_id:
            raise ValueError("Run ID is empty")

        detail = fetch_run_detail(run_id, api_key)
        info_md = f"## 📋 Training Run Details: `{run_id}`\n\n"
        if detail:
            for k, v in detail.items():
                if k != "metrics":
                    info_md += f"- **{k.replace('_', ' ').title()}**: `{v}`\n"
        else:
            info_md += "⚠️ No details available for this run."

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=info_md),
        )

    except Exception as e:
        error_msg = f"❌ Failed to load details: {str(e)}"
        print(error_msg)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=error_msg),
        )


# ====== Gradio App ======
with gr.Blocks(title="TuFT Platform") as demo:
    gr.Markdown("# TuFT Platform")
    gr.Markdown("*(All functionality is simulated — no real training or inference occurs)*")

    # States
    api_keys_state = gr.State([])  # List[str]
    selected_api_key_state = gr.State("")  # str
    models_state = gr.State([])  # List[str]
    sampling_model_type_state = gr.State("Base Model")  # Track current model type in Sampling tab

    with gr.Row():
        with gr.Column():
            api_key_dropdown = gr.Dropdown(
                label="Select API Key", choices=[], value=None, allow_custom_value=True
            )
            with gr.Row():
                set_key_btn = gr.Button("Set API Key")
                create_key_btn = gr.Button("Create New API Key")

    # Navigation state
    current_view = gr.State("resources")

    with gr.Row():
        # ====== LEFT SIDEBAR (Navigation) ======
        with gr.Column(scale=1, min_width=180):
            gr.Markdown("## Navigation")
            btn_resources = gr.Button("Resources")
            btn_training = gr.Button("Training Runs")
            btn_checkpoints = gr.Button("Checkpoints")
            btn_sampling = gr.Button("Sampling")

        # ====== RIGHT CONTENT AREA ======
        with gr.Column(scale=4):
            # ========== Panel 1: Resources (API Keys + Dataset Upload) ==========
            with gr.Column(visible=True) as panel_resources:
                gr.Markdown("## 🔑 API Keys")
                api_key_display = gr.Textbox(label="Current API Key", interactive=False)
                api_keys_df = gr.Dataframe(
                    headers=["API Key"],
                    value=pd.DataFrame({"API Key": []}),
                    interactive=False,
                )

            # ========== Panel 2: Training Runs ==========
            with gr.Column(visible=False) as panel_training_list:
                gr.Markdown("## 🏃 Training Runs")
                runs_df = gr.Dataframe(
                    headers=["ID", "BASE MODEL", "LORA RANK", "LAST REQUEST TIME"],
                    value=pd.DataFrame(),
                    interactive=False,
                )

            # ========== Panel 2.1: Training Run Detail ==========
            with gr.Column(visible=False) as panel_training_detail:
                detail_info = gr.Markdown()
                back_to_runs_btn = gr.Button("⬅️ Back to Runs")
            runs_df.select(
                fn=show_training_detail,
                inputs=[runs_df, selected_api_key_state],
                outputs=[panel_training_list, panel_training_detail, detail_info],
            )
            # ========== Panel 3: Checkpoints ==========
            with gr.Column(visible=False) as panel_checkpoints:
                gr.Markdown("## 💾 Checkpoints")
                ckpt_df = gr.Dataframe(
                    headers=["ID", "TYPE", "PATH", "SIZE", "VISIBILITY", "CREATED"],
                    value=pd.DataFrame(),
                    interactive=False,
                )

            # ========== Panel 4: Sampling ==========
            with gr.Column(visible=False) as panel_sampling:
                gr.Markdown("## 🧪 Sampling")

                # Step 1: Choose model type
                model_type = gr.Radio(
                    choices=["Base Model", "Fine-tuned Model"],
                    value=None,
                    label="Model Type",
                )

                # Sync model_type to state
                model_type.change(lambda x: x, inputs=model_type, outputs=sampling_model_type_state)

                # Step 2: Two dropdowns, only one visible at a time
                base_model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Base Model",
                    interactive=False,
                    visible=False,
                )
                ft_model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Fine-tuned Model (Checkpoint)",
                    interactive=False,
                    visible=False,
                )

                # Unified toggle function using update_model_choices
                def toggle_model_ui(model_type_val, api_key):
                    if not model_type_val:
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    base_models, ft_models = get_model_choices(model_type_val, api_key)

                    if model_type_val == "Base Model":
                        return (
                            gr.update(
                                choices=base_models,
                                value=None,
                                interactive=bool(base_models),
                                visible=True,
                            ),
                            gr.update(visible=False),
                        )
                    elif model_type_val == "Fine-tuned Model":
                        return (
                            gr.update(visible=False),
                            gr.update(
                                choices=ft_models,
                                value=None,
                                interactive=bool(ft_models),
                                visible=True,
                            ),
                        )
                    else:
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                model_type.change(
                    fn=toggle_model_ui,
                    inputs=[model_type, selected_api_key_state],
                    outputs=[base_model_dropdown, ft_model_dropdown],
                )

                with gr.Accordion("Sampling Parameters", open=True):
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness: lower = more deterministic, "
                            "higher = more creative",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p (Nucleus Sampling)",
                            info="Cumulative probability threshold; 1.0 = no filtering",
                        )

                    with gr.Row():
                        top_k = gr.Slider(
                            minimum=-1,
                            maximum=100,
                            value=-1,
                            step=1,
                            label="Top-k",
                            info="Number of highest-probability tokens to keep; -1 = disable",
                        )
                        max_tokens = gr.Slider(
                            minimum=10,
                            maximum=2048,
                            value=256,
                            step=10,
                            label="Max Tokens",
                            info="Maximum number of tokens to generate",
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        stop_sequences = gr.Textbox(
                            placeholder='e.g., "\\n\\n", "###", "</s>"',
                            label="Stop Sequences",
                            info="Comma-separated strings to stop generation (e.g., '\\n\\n, END')",
                            interactive=True,
                        )
                        seed = gr.Number(
                            value=None,
                            label="Seed",
                            info="Integer for reproducible output; leave empty for random",
                            precision=0,
                        )

                input_text = gr.Textbox(
                    lines=3, placeholder="Enter your prompt...", label="Input Text"
                )
                sample_btn = gr.Button("Generate Sample", variant="primary")
                sample_output = gr.Textbox(label="Sampling Result", lines=10, interactive=False)

                # Helper function to parse stop sequences
                def parse_stop_sequences(stop_str: str):
                    if not stop_str:
                        return None
                    sequences = [s.strip() for s in stop_str.split(",") if s.strip()]
                    return sequences if sequences else None

                # Sampling function
                def real_sampling(
                    model_type_val,
                    base_model_val,
                    ft_model_val,
                    temp,
                    top_p_val,
                    top_k_val,
                    max_tokens_val,
                    input_txt,
                    api_key,
                    stop_seq_input,
                    seed_input,
                ):
                    if not input_txt or not input_txt.strip():
                        return "❌ Please provide input text."

                    if model_type_val == "Base Model":
                        if not base_model_val:
                            return "❌ Please select a base model."
                        payload = {
                            "data_list": [input_txt.strip()],
                            "base_model": base_model_val,
                            "model_path": None,
                        }
                    else:
                        if not ft_model_val:
                            return "❌ Please select a fine-tuned model."
                        payload = {
                            "data_list": [input_txt.strip()],
                            "base_model": None,
                            "model_path": ft_model_val,
                        }

                    payload.update(
                        {
                            "temperature": temp,
                            "top_p": top_p_val,
                            "top_k": int(top_k_val) if top_k_val != -1 else -1,
                            "max_tokens": int(max_tokens_val),
                            "seed": (int(seed_input) if seed_input not in (None, "") else None),
                            "stop": parse_stop_sequences(stop_seq_input),
                        }
                    )
                    sample_result = run_sample(payload, api_key)
                    cleaned_results = re.sub(r"(?s)<think>\n.*?</think>\n\n", "", sample_result)
                    if "<think>" in cleaned_results:
                        return "❌ Invalid response: please increase the max tokens"
                    return cleaned_results

                sample_btn.click(
                    fn=real_sampling,
                    inputs=[
                        model_type,
                        base_model_dropdown,
                        ft_model_dropdown,
                        temperature,
                        top_p,
                        top_k,
                        max_tokens,
                        input_text,
                        selected_api_key_state,
                        stop_sequences,
                        seed,
                    ],
                    outputs=sample_output,
                )

            # ========== Navigation Logic ==========
            def switch_to_resources():
                return [
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                ]

            def switch_to_training(selected_api_key):
                runs = load_runs(selected_api_key) if selected_api_key else []
                df = (
                    pd.DataFrame(runs)
                    if runs
                    else pd.DataFrame(
                        columns=["ID", "BASE MODEL", "LORA RANK", "LAST REQUEST TIME"]
                    )
                )
                return [
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=df),
                ]

            def switch_to_checkpoints(selected_api_key):
                ckpts = load_ckpts(selected_api_key) if selected_api_key else []
                df = (
                    pd.DataFrame(ckpts)
                    if ckpts
                    else pd.DataFrame(
                        columns=["ID", "TYPE", "PATH", "SIZE", "VISIBILITY", "CREATED"]
                    )
                )
                return [
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value=df),
                ]

            def switch_to_sampling(current_model_type, api_key):
                base_vis = False
                ft_vis = False
                base_choices = []
                ft_choices = []

                if current_model_type == "Base Model":
                    base_vis = True
                    base_choices = load_models(api_key) if api_key else []
                elif current_model_type == "Fine-tuned Model":
                    ft_vis = True
                    ckpts = load_ckpts(api_key) if api_key else []
                    ft_choices = [ckpt.get("path", "") for ckpt in ckpts if "path" in ckpt]

                return [
                    gr.update(visible=False),  # panel_resources
                    gr.update(visible=False),  # panel_training_list
                    gr.update(visible=False),  # panel_training_detail
                    gr.update(visible=False),  # panel_checkpoints
                    gr.update(visible=True),  # panel_sampling
                    gr.update(
                        choices=base_choices,
                        value=None,
                        interactive=bool(base_choices),
                        visible=base_vis,
                    ),
                    gr.update(
                        choices=ft_choices,
                        value=None,
                        interactive=bool(ft_choices),
                        visible=ft_vis,
                    ),
                ]

            btn_resources.click(
                switch_to_resources,
                outputs=[
                    panel_resources,
                    panel_training_list,
                    panel_training_detail,
                    panel_checkpoints,
                    panel_sampling,
                ],
            )
            btn_training.click(
                switch_to_training,
                inputs=[selected_api_key_state],
                outputs=[
                    panel_resources,
                    panel_training_list,
                    panel_training_detail,
                    panel_checkpoints,
                    panel_sampling,
                    runs_df,
                ],
            )
            btn_checkpoints.click(
                switch_to_checkpoints,
                inputs=[selected_api_key_state],
                outputs=[
                    panel_resources,
                    panel_training_list,
                    panel_training_detail,
                    panel_checkpoints,
                    panel_sampling,
                    ckpt_df,
                ],
            )
            btn_sampling.click(
                switch_to_sampling,
                inputs=[sampling_model_type_state, selected_api_key_state],
                outputs=[
                    panel_resources,
                    panel_training_list,
                    panel_training_detail,
                    panel_checkpoints,
                    panel_sampling,
                    base_model_dropdown,
                    ft_model_dropdown,
                ],
            )

            back_to_runs_btn.click(
                lambda: (gr.update(visible=True), gr.update(visible=False)),
                outputs=[panel_training_list, panel_training_detail],
            )

    # ====== API Key Management ======

    def set_api_key(api_key_input, api_keys):
        if not isinstance(api_key_input, str) or not api_key_input.strip():
            current_key = ""
        else:
            current_key = api_key_input.strip()

        if not current_key:
            return (
                api_keys,
                "",
                gr.Dropdown(choices=api_keys, value=None),
                gr.Dataframe(value=[[k] for k in api_keys], headers=["API Keys"])
                if api_keys
                else gr.Dataframe(value=[], headers=["API Keys"]),
                "",
            )

        # If it's a new key (not in list), add it
        if current_key not in api_keys:
            api_keys = api_keys + [current_key]

        return (
            api_keys,
            current_key,
            gr.Dropdown(choices=api_keys, value=current_key),
            gr.Dataframe(value=[[k] for k in api_keys], headers=["API Keys"]),
            current_key,
        )

    def create_new_api_key(api_keys):
        new_key = generate_api_key()
        updated_keys = api_keys + [new_key]
        return (
            updated_keys,  # api_keys_state
            new_key,  # selected_api_key_state
            gr.Dropdown(choices=updated_keys, value=new_key),  # api_key_dropdown
            gr.Dataframe(
                value=[[k] for k in updated_keys], headers=["API Keys"]
            ),  # api_keys_df (adjust if your df format differs)
            new_key,  # api_key_display (or whatever you use to show it)
        )

    create_key_btn.click(
        create_new_api_key,
        inputs=[api_keys_state],
        outputs=[
            api_keys_state,
            selected_api_key_state,
            api_key_dropdown,
            api_keys_df,
            api_key_display,
        ],
    )

    set_key_btn.click(
        set_api_key,
        inputs=[api_key_dropdown, api_keys_state],
        outputs=[
            api_keys_state,
            selected_api_key_state,
            api_key_dropdown,
            api_keys_df,
            api_key_display,
        ],
    )

    def on_api_key_selected(selected_key: str, api_keys: List[str]):
        if not selected_key or selected_key not in api_keys:
            return [], selected_key
        models = load_models(selected_key)
        return models, selected_key

    # 🔥 CRITICAL FIX: When API key changes, update Sampling model dropdowns
    def update_sampling_models_on_api_key_change(api_key: str, current_model_type: str):
        if not current_model_type:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )

        base_models, ft_models = get_model_choices(current_model_type, api_key)

        if current_model_type == "Base Model":
            return (
                gr.update(
                    choices=base_models,
                    value=None,
                    interactive=bool(base_models),
                    visible=True,
                ),
                gr.update(visible=False),
            )
        elif current_model_type == "Fine-tuned Model":
            return (
                gr.update(visible=False),
                gr.update(
                    choices=ft_models,
                    value=None,
                    interactive=bool(ft_models),
                    visible=True,
                ),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )

    api_key_dropdown.change(
        on_api_key_selected,
        inputs=[api_key_dropdown, api_keys_state],
        outputs=[models_state, selected_api_key_state],
    ).then(
        fn=update_sampling_models_on_api_key_change,
        inputs=[selected_api_key_state, sampling_model_type_state],
        outputs=[base_model_dropdown, ft_model_dropdown],
    )
# Launch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio app with custom port")
    parser.add_argument(
        "--port",
        type=int,
        default=10612,
        help="Port to run the console on (default: 10612)",
    )
    parser.add_argument(
        "--backend_port",
        type=int,
        default=10710,
        help="Port to run the server on (default: 10710)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    console_config.CONSOLE_SERVER_URL = f"http://127.0.0.1:{args.backend_port}/api/v1"

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port)
