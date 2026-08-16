name: Feature Request

description: Suggest a new feature or improvement
title: "[Feature] "
labels: [enhancement]
body:
  - type: markdown
    attributes:
      value: |
        ## Feature Description
        A clear description of the feature or improvement you are proposing.

  - type: textarea
    id: motivation
    attributes:
      label: "Motivation"
      description: "Why is this feature useful? What problem does it solve?"
    validations:
      required: true

  - type: textarea
    id: proposal
    attributes:
      label: "Proposed Solution"
      description: "How would you like it to work?"
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: "Alternatives Considered"
      description: "Any other approaches you considered"

  - type: textarea
    id: additional
    attributes:
      label: "Additional Context"
      description: "Screenshots, mockups, or other relevant information"
