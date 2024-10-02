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
        * Endpoint that streams response tokens back
    * `/generate`
        * Endpoint that generates the response and returns it in the response, no streaming

# Technical Default Implementations
* Inference:
    * We assume an attention mask that allows all tokens to be attended to autoregressively!
        * Any other masking implementations, like sliding window, other other memory optimizations, are left as an exercise to the inferencer
    * We cache attention keys and values 
        * If you'd like to make memory optimizations, feel free to tinker with this or disable!

# Other Notes and Considerations
* You can inference each endpoint locally in `use.py` to ensure proper functionality and testing!
* We use a CUDA GPU if it's available, otherwise defaults to the CPU

# TODOs:
* Full API Features in accordance with [Llama Stack](https://github.com/meta-llama/llama-stack)
* fine tuning support
* .gguf support
* VRAM / RAM warning if size incompatible
