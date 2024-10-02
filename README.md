# Cerebral Valley Compute Boilerplate
Quick and easy endpoints to inference any Huggignface hosted LLM. Simply clone this repo, edit your config, and spin up an API!

# Steps to Run
1. Clone this repository on the machine you're spinning up the API on.
2. Install all dependencies with `pip3 install -r requirements.txt`.
3. Edit `constants.py` to your desired configuration.
4. Run `python3 main.py` to download the model and start the endpoint (*NOTE: downloading the LLM, depending on the size, may take a while*).

# Current Features
* Inference any Huggingface LLM:
    * `/generate-stream`
    * API endpoint that streams tokens back from your configured they are inferenced

# Other Notes and Considerations
* There is a test function, `test.py`, for you to use locally and ensure your model is inferencing properly!
* We use a CUDA GPU if it's available, otherwise defaults to the CPU
* We assume an attention mask that allows all tokens to be attended to autoregressively!
    * Any other masking implementations, like sliding window, other other memory optimizations, are left as an exercise to the inferencer

# TODOs:
* Full API Features in accordance with [Llama Stack](https://github.com/meta-llama/llama-stack)
* fine tuning support
* .gguf support
* VRAM / RAM warning if size incompatible
