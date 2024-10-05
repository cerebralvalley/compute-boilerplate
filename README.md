# Cerebral Valley Compute Boilerplate
Turn any machine into a suite of inference and tuning endpoints for Huggingface LLMs. Simply clone this repo, edit your config, and spin up an API!

# Steps to Run
1. Clone this repository on the machine you're spinning up the API on.
2. Install all dependencies with `pip3 install -r requirements.txt`.
3. Edit `config.py` to your desired configuration.
4. Ensure that the API's port is open on your machine. If you're running on a Nebius machine, this port should already be open. If you're running on AWS, you will need to manually whitelist the port. 
5. Run `python3 main.py` to download the model and start the endpoint (*NOTE: downloading the LLM, depending on the size, may take a while*).

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
* In `scripts/setup/sh`, we've provided a script to setup and install all necessary dependencies
* For the `Config` parameter `QUANTIZATION_BITS`: this is only used to calculate the model's size, and must be manually input for an accurate estimate
    * First, we attempt to estimate the model's size from Pytorch, and if we can't, then we fallback to this parameter

# SSH Commands
* Creating a Key:
  ```
  ssh-keygen -t ed25519
  ```
  Folder: `/path/to/your/keys/folder/KEY_NAME`

* Copying the public key:
  ```
  cat /path/to/your/keys/folder/KEY_NAME.pub
  ```

* Connecting to the server:
  ```
  ssh -i /path/to/your/keys/folder/KEY_NAME <USERNAME>@<SERVER_IP>
  ```

Note: Replace `/path/to/your/keys/folder/KEY_NAME`, `username`, and `server_ip` with your actual values.

# TODOs:
* Full API Features in accordance with [Llama Stack](https://github.com/meta-llama/llama-stack)
* fine tuning support
* .gguf support
