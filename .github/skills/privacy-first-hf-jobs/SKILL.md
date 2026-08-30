---
name: privacy-first-hf-jobs
description: 'Launch, monitor, and clean up Hugging Face Jobs while minimizing source and checkpoint exposure. Use for HF Jobs, cloud training, organization credits, job naming, artifact storage, secrets, privacy checks, and local job tracking.'
argument-hint: 'Describe the training job to launch or audit'
---

# Privacy-first Hugging Face Jobs

Use this workflow for every Hugging Face Job associated with this workspace.

## Security boundary

A Job launched under an organization is auditable by that organization. Its existence, initiator, hardware, command, status, billing, and possibly logs may be visible. Never promise that a Job is anonymous, untrackable, or hidden from administrators. Do not disguise its purpose with deceptive or random names.

Use concise, non-sensitive, truthful names such as `hc-train-YYYYMMDD-HHMM`. Names must not contain secrets, private dataset names, or detailed research claims.

## Preflight

1. Confirm the billing namespace and explain who can see Job metadata.
2. Confirm the source and output repositories are owned by the user's personal namespace and are private.
3. Never pass tokens, credentials, or private URLs as ordinary environment variables or command arguments. Use Job secrets.
4. Never submit a local directory as a volume to an organization Job. Local volumes are copied into the organization's `jobs-artifacts` bucket.
5. Never mount an organization bucket for source or output unless the user explicitly accepts organization-member access.
6. Prefer a prebuilt private image or downloading a release archive from a private personal repository inside the ephemeral container.
7. Upload checkpoints periodically to a private personal model repository so cancellation does not lose progress.
8. Record the Job locally with [track-job.sh](./scripts/track-job.sh). Never put secrets in the tracker.

## Launch review

Before launching, display and verify:

- Namespace and billing owner
- Honest, non-sensitive Job name
- GPU flavor and timeout
- Exact command, with secrets redacted
- Every mounted volume and its owner
- Source repository visibility
- Checkpoint destination visibility
- Recovery behavior if the Job is canceled

Block the launch if it includes a local-directory volume, an organization-owned artifact destination, plaintext credentials, or an unverified private repository.

## Monitoring

Track the Job ID, status, checkpoint iteration, and timestamp locally. Monitoring must not expose secret values. Organization administrators remain able to audit usage.

## Cleanup

1. Verify the newest checkpoint exists locally and in the private personal repository.
2. Verify the repository reports `private: true`.
3. Cancel or allow the Job to finish.
4. Identify exact artifact prefixes from that Job's volume metadata.
5. Delete only prefixes owned by this project; never delete the shared bucket or another member's files.
6. Verify the prefixes are absent.
7. Record the terminal state in the local tracker.
8. Explain that deletion cannot revoke previous downloads or guarantee deletion from provider backups and audit records.
