SPHINXOPTS    ?=
SPHINXBUILD   ?= $(if $(wildcard .venv/bin/sphinx-build),.venv/bin/sphinx-build,uv run sphinx-build)
SOURCEDIR_EN  = source
SOURCEDIR_ZH  = source_zh
BUILDDIR      = build
VERSION       ?= latest
OUT_EN        = $(BUILDDIR)/html/en/$(VERSION)
OUT_ZH        = $(BUILDDIR)/html/zh/$(VERSION)
PYTHON        ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
SWITCHER_JSON := switcher.json
PROJECT_VERSION := $(shell $(PYTHON) -c "import json; items=json.load(open('$(SWITCHER_JSON)','r',encoding='utf-8')); vals=[i['version'] for i in items if i.get('version')!='latest']; print(vals[0] if vals else '')")

.PHONY: help clean html html-en html-zh switcher

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html      to build HTML files for both languages"
	@echo "  html-en   to build HTML files for English only"
	@echo "  html-zh   to build HTML files for Chinese only"
	@echo "  clean     to remove build directory"

clean:
	rm -rf $(BUILDDIR)

html: html-en html-zh
	@$(MAKE) switcher

html-en:
	@echo "Building English documentation..."
	$(SPHINXBUILD) -b html $(SOURCEDIR_EN) $(OUT_EN) $(SPHINXOPTS)
	@echo "English documentation built."

html-zh:
	@echo "Building Chinese documentation..."
	$(SPHINXBUILD) -b html $(SOURCEDIR_ZH) $(OUT_ZH) $(SPHINXOPTS)
	@echo "Chinese documentation built."

switcher:
	@echo "Writing switcher.json..."
	cp $(SWITCHER_JSON) $(BUILDDIR)/html/
	@if [ "$(VERSION)" = "latest" ] && [ -n "$(PROJECT_VERSION)" ]; then \
		ln -sfn latest $(BUILDDIR)/html/en/$(PROJECT_VERSION); \
		ln -sfn latest $(BUILDDIR)/html/zh/$(PROJECT_VERSION); \
	fi
