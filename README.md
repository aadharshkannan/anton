# Anton

A repository that enables anyone to create their own LLM from WhatsApp data.

## Directory Structure

```text
anton/
├── data/
│   ├── raw/                     # Raw WhatsApp export
│   ├── processed/               # Cleaned and structured chat data
│   └── hellaswag_format/        # Final dataset formatted like HellaSwag
│
├── scripts/
│   ├── parse_whatsapp.py        # Extract & clean WhatsApp history
│   ├── shared_models.py         # Pydantic models that store structured data
│   ├── generate_response.py     # Generate adversarial response generation using LLM
│   └── fine-tune-example.ipynb  # Script to fine-tune the LLM
│
├── models/
│   └── checkpoints/             # Save fine-tuned models here
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview and usage instructions
└── .gitignore                   # Files/directories to ignore in version control
