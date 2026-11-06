# Thin wrappers around the `persona` CLI. Examples:
#   make data PERSONA=juniper LIMIT=20
#   make train PERSONA=juniper SET="train.base_model=Qwen/Qwen2.5-0.5B-Instruct train.max_steps=30"
PERSONA ?= juniper
LIMIT   ?=
SET     ?=
PY      ?= python

ARGS = $(PERSONA) $(if $(LIMIT),--limit $(LIMIT)) $(foreach s,$(SET),--set $(s))

.PHONY: install install-gpu test validate data dpo-data train train-dpo eval capability chat all

install:          ; pip install -e '.[dev]'
install-gpu:      ; pip install -e '.[train,dev]' && pip install unsloth
test:             ; $(PY) -m pytest -q
validate:         ; persona validate $(ARGS)
data:             ; persona data $(ARGS)
dpo-data:         ; persona dpo-data $(ARGS)
train:            ; persona train-sft $(ARGS)
train-dpo:        ; persona train-dpo $(ARGS)
eval:             ; persona eval $(ARGS)
capability:       ; persona capability $(ARGS)
chat:             ; persona chat $(ARGS)
all: data train eval
