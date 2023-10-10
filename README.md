# flask-trigger-detection
Flask based web application that detects the presence of triggering content in a given long document.

I coded the Hierarchical Recurrence concept over HuggingFace's transformer-based encoder in Pytorch. The details of the implementation and method can be found in the [paper](https://arxiv.org/abs/2307.14912).

Don't forget to supply your own Huggingface API access token when running the code. You can do it by adding .env file to the root directory of the project and putting your token as follows:
```
hf_key = "YOUR_ACCESS_TOKEN"
```



