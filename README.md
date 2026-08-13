# AI-Powered Student Wellness Triage System

**Hackathon Prototype | Python • FastAPI • NLP • Machine Learning • REST APIs**

An independently developed student hackathon prototype exploring how natural-language processing and machine-learning techniques can be used to analyze written student wellness concerns and route simulated cases through predefined support workflows.

The project combines a FastAPI backend, transformer-based NLP classification, REST API endpoints, demo user workflows, and a dashboard for reviewing simulated student submissions.

> **Prototype Notice:** This project was created for educational and software-demonstration purposes. It is not a clinical system, medical device, emergency-response service, or production healthcare application.

## Project Goals

The project was designed to explore several areas of software engineering and applied artificial intelligence:

* Building and documenting REST APIs with FastAPI
* Applying NLP and transformer-based models to text classification
* Combining multiple classification signals into an application workflow
* Designing role-based user interactions
* Creating alert and review workflows
* Practicing authentication and protected-data-handling concepts
* Testing application behavior and performance
* Integrating machine-learning components into a larger software system

## Key Features

### NLP and Machine Learning

* Analyzes student-written text using NLP classification models
* Uses transformer-based models through Hugging Face
* Evaluates text for predefined wellness-related categories and indicators
* Combines model outputs to support prototype triage decisions
* Demonstrates integration of trained or fine-tuned models into an application backend

### FastAPI Backend

* REST API architecture implemented with FastAPI
* API endpoints for submitting and processing simulated student data
* Structured request and response handling
* API documentation generated through FastAPI
* Modular backend design supporting application and model components

### Demonstration Workflows

* Simulated student-submission workflow
* Demo roster-validation functionality
* Role-based workflows for reviewing submitted information
* Dashboard for viewing and managing simulated cases
* Prototype alert and notification logic

These workflows are intended only to demonstrate application design and software functionality.

## Security Concepts

The application includes prototype implementations of several security-related concepts, including:

* Authentication
* Role-based access
* Protected-data-handling concepts
* Controlled access to demonstration workflows
* Storage and retrieval of prototype alert information

These features were implemented to explore software-security concepts within a student project.

They should **not** be interpreted as production-grade security controls or as evidence that the application meets healthcare, institutional, regulatory, or industry compliance requirements.

## Privacy, Security, and HIPAA Disclaimer

This project is a **student hackathon prototype developed for educational and demonstration purposes**.

It is **not HIPAA compliant or HIPAA certified, and no claim of HIPAA compliance is made**.

The application was not designed, reviewed, audited, or deployed as a production system for handling protected health information.

A real-world application involving sensitive health information would require substantially more work, including appropriate:

* Security architecture
* Encryption and key-management practices
* Identity and access management
* Privacy controls
* Audit logging
* Infrastructure security
* Data-retention policies
* Incident-response procedures
* Legal and regulatory review
* Organizational policies and procedures
* Security testing and validation

Those requirements are outside the scope of this hackathon project.

## Data and Demonstration Environment

The project is intended to operate with **synthetic, sample, or demonstration data**.

Any included users, accounts, scenarios, records, or workflows should be treated as test data created for software development and demonstration.

The repository should not be used to store or process actual protected health information, confidential student records, or real clinical information.

## Clinical and Emergency-Use Disclaimer

This application does **not**:

* Diagnose mental-health conditions
* Provide medical or psychological advice
* Determine whether a person is experiencing a medical or psychiatric emergency
* Replace qualified healthcare or mental-health professionals
* Replace institutional crisis-response procedures
* Replace emergency services

Model outputs are generated as part of a software prototype and should not be interpreted as clinical judgments.

## Technology Stack

**Languages**

* Python

**Backend**

* FastAPI
* REST APIs

**AI / Machine Learning**

* Natural Language Processing
* Transformer-based text classification
* Hugging Face models

**Data Processing**

* pandas
* Supporting Python data-processing libraries

**Software Engineering**

* API design
* Modular application architecture
* Testing
* Load testing
* Technical documentation

**Security Concepts**

* Authentication
* Role-based access
* Protected-data-handling concepts
* Prototype alert-storage workflows

## Datasets

The project explored publicly available datasets for model development and experimentation, including:

* Dreaddit Stress Analysis dataset
* Dartmouth StudentLife dataset

Dataset use was part of the project's educational machine-learning experimentation and should not be interpreted as validation of the system for clinical or production use.

## Testing

The project includes testing and load-testing utilities intended to evaluate prototype application behavior, API functionality, and basic performance characteristics.

Testing performed for this project does not constitute:

* Security certification
* Regulatory validation
* Clinical validation
* Production-readiness testing

## What This Project Demonstrates

This repository is intended primarily as a software-development portfolio project demonstrating experience with:

* Python development
* FastAPI
* REST API development
* NLP and transformer-based machine learning
* Hugging Face model integration
* Data processing
* API and application testing
* Modular software design
* Technical documentation
* Authentication and access-control concepts
* Integrating AI components into an end-to-end application

## Project Context

This project was independently developed during a weeklong AI Club hackathon.

The objective was to build a functioning proof of concept that combined software engineering, machine learning, API development, testing, and application design within a limited development period.

The resulting system should be evaluated as a **student software prototype and portfolio project**, not as a production healthcare platform.

## Responsible Use

This repository is provided for educational, demonstration, and portfolio purposes.

Do not use this software to make clinical decisions, evaluate real individuals, provide emergency-response services, or process protected or confidential health information without appropriate professional, legal, security, privacy, and institutional review.
