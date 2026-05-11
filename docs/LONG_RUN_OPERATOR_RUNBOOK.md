# Long-Run Operator Runbook

Operational guide for executing multi-hour Hybrid-ALife campaigns safely. Covers preflight, environment, session management, recovery, and exact command recipes.

> Work in progress — this document is being expanded in the `sprint/operator-runbook-fast` branch. See sections below.

## Table of Contents

1. [Preflight Checklist](#preflight-checklist)
2. [JAX & Cache Environment](#jax--cache-environment)
3. [Session Management (tmux / nohup)](#session-management-tmux--nohup)
4. [Resume vs. Force Policy](#resume-vs-force-policy)
5. [Worker Selection](#worker-selection)
6. [Disk Hygiene](#disk-hygiene)
7. [Failure Recovery](#failure-recovery)
8. [When to Stop a Bad Campaign](#when-to-stop-a-bad-campaign)
9. [Artifact Packaging](#artifact-packaging)
10. [Command Recipes](#command-recipes)

