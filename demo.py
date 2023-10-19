import gradio as gr

import uvd


def proc_video(video, preprocessor_name):
    subgoals = uvd.get_uvd_subgoals(video, preprocessor_name.lower().replace("-", ""))
    return gr.Gallery(
        value=[(img, f"No. {i+1} subgoal") for i, img in enumerate(subgoals)]
    )


with gr.Blocks() as demo:
    with gr.Row():
        input_video = gr.Video(height=224, width=224, scale=3)
        preprocessor_name = gr.Dropdown(
            ["VIP", "R3M", "LIV", "CLIP", "DINO-v2", "VC-1", "ResNet"],
            label="Preprocessor",
            value="VIP",
            height=224,
            width=56,
            scale=1,
        )
        output = gr.Gallery(label="UVD SubGoals", height=224, preview=True, scale=4)
    with gr.Row():
        submit = gr.Button("Submit")
        clr = gr.ClearButton(components=[input_video, output])
    submit.click(proc_video, inputs=[input_video, preprocessor_name], outputs=[output])


demo.queue().launch(share=True, show_error=True)
