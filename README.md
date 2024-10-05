# Cerebral Valley Compute Boilerplate
Turn any machine into a suite of inference and tuning endpoints for Huggingface LLMs. Simply clone this repo, edit your config, and spin up an API!

# Steps to Run
1. Clone this repository on the machine you're spinning up the API on.
2. Install all dependencies with `pip3 install -r requirements.txt`.
3. Edit `config.py` to your desired configuration.
4. Ensure that the API's port is open on your machine. If you're running on a Nebius machine, this port should already be open. If you're running on AWS, you will need to manually whitelist the port. 
5. If your Huggingface model you want to inference is access gated, ensure you pass in a token in `.env` to allow authentication. Your token should be fine grained to allow access to gated repositories.
6. Run `python3 main.py` to download the model and start the endpoint (*NOTE: downloading the LLM, depending on the size, may take a while*).

# Current Features
* Checks disk size, vRAM, and RAM for model compatibility
* Fully compatible with the [Llama Stack](https://github.com/meta-llama/llama-stack)!
* Inference any Huggingface LLM:
    * `/inference/completion`
        * Endpoint that generates a completion response
        * Supports both streaming and non-streaming modes
    * `/inference/chat_completion`
        * Endpoint that generates a chat completion response
        * Supports both streaming and non-streaming modes

# Default Technical Implementations
* Inference:
    * We assume an attention mask that allows all tokens to be attended to autoregressively!
        * Any other masking implementations, like sliding window, other other memory optimizations, are left as an exercise to the inferencer
    * We cache attention keys and values 
        * If you'd like to make memory optimizations, feel free to tinker with this or disable!
    * The default temperature is set to 1, meaning we just select the highest probability token without any distribution adjustment. If any other temperature is set, we apply the temperature to the logits then sample from the resulting softmax distribution
      * FYI, temperature is used to adjust the probability distribution for next token selection.
      * Temperature of 1.0 keeps the original distribution, < 1.0 makes it more peaked, > 1.0 makes it more uniform.

# Notes and Considerations
* You can inference each endpoint locally in `use.py` to ensure proper functionality and testing!
* We use a CUDA GPU if it's available, otherwise defaults to the CPU
* In `scripts/setup/sh`, we've provided a script to setup and install all necessary dependencies
* Some models require an agreement or signature to access. For these models, please sign the access documents on the model's page, and then input your Huggingface Access Token in `.env` to override this
* To download and inference models, we need BOTH sufficient *disk* and *memory* space
    * **Before the server starts, a check will be made to ensure this is the case**
    * Huggingface first downloads the weights to disk in a cache folder, which is then loaded into memory upon inference with the `.to(device)` function
    * If you run out of disk space on accident, huggingface downloads (on UNIX systems) to `~/.cache/huggingface/hub` that you can clear

# SSH Commands
If you're using a virtual machine, here are some helpful SSH commands to get you started.
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

# Bug and Fix Log
* `json.decoder.JSONDecodeError: Extra data: line 1 column 55 (char 54)` error on reading of a stream
  * This is because there is no deliminattion between stream chunks (we are using `requests` `read_lines()` function, which auto separates a streamed response by `\n`)
  * To fix this, we add a bytestring `b'\n'` to the end of the yielded chunk in the API

# TODOs:
* fine tuning support
* .gguf support
