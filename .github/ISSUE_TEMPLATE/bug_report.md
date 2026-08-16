name: Bug Report

description: Report something that is not working as expected
title: "[Bug] "
labels: [bug]
body:
  - type: markdown
    attributes:
      value: |
        ## Bug Description
        A clear description of the bug.

  - type: textarea
    id: steps
    attributes:
      label: "Steps to Reproduce"
      description: "How to reproduce the issue"
      placeholder: |
        1. Run ...
        2. See error ...
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: "Expected Behavior"
      description: "What should happen"
    validations:
      required: true

  - type: textarea
    id: actual
    attributes:
      label: "Actual Behavior"
      description: "What actually happens"
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: "Environment"
      description: "Python version, OS, installed packages"
      placeholder: |
        - Python: 3.x
        - OS: Ubuntu 22.04
        - Packages: torch 2.x, ...
    validations:
      required: true

  - type: textarea
    id: traceback
    attributes:
      label: "Full Traceback (if applicable)"
      description: "Paste the full error traceback here"

  - type: textarea
    id: extra
    attributes:
      label: "Additional Context"
      description: "Screenshots, logs, or other relevant information"
