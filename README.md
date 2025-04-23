# anton
Repository that allows anyone to create their own LLM
anton/
├── data/
│   ├── raw/                    # Raw WhatsApp export
│   ├── processed/              # Cleaned and structured chat data
│   └── hellaswag_format/      # Final dataset formatted like HellaSwag
│
├── scripts/
│   ├── parse_whatsapp.py      # Extract & clean WhatsApp history
│   ├── create_prompts.py      # Generate LLM prompts for adversarial response generation
│   ├── generate_responses.py  # Use LLM to generate adversarial responses
│   ├── format_hellaswag.py    # Convert responses to HellaSwag format
│   └── fine_tune.py           # Script to fine-tune the LLM
│
├── models/
│   └── checkpoints/           # Save fine-tuned models here
│
├── evaluation/
│   └── evaluate_model.py      # Evaluate model accuracy / adversarial resistance
│
├── config/
│   └── model_config.yaml      # Parameters for model, training, etc.
│
├── requirements.txt
├── README.md
└── .gitignore
