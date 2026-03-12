/**
 * Owner mapping and access control for memory-lancedb-brain
 */

import type { MemoryScope } from "./schema.js";

export interface OwnerConfig {
  owner_id: string;
  owner_namespace: string;
  channels?: Record<string, string>;
}

export interface ResolvedOwner {
  ownerId: string;
  ownerNamespace: string;
}

export interface AccessResult {
  allowed: boolean;
  reason?: string;
}

export interface ResolveOwnerInput {
  senderId?: string;
  messageChannel?: string;
  agentId?: string;
  senderIsOwner?: boolean;
}

export function normalizeOwners(owners?: OwnerConfig[]): OwnerConfig[] {
  return (owners ?? []).map((owner) => ({
    owner_id: owner.owner_id,
    owner_namespace: owner.owner_namespace,
    channels: owner.channels ?? {},
  }));
}

export function resolveOwnerFromContext(
  input: ResolveOwnerInput,
  owners: OwnerConfig[],
): ResolvedOwner | null {
  const normalizedOwners = normalizeOwners(owners);
  const { senderId, messageChannel, senderIsOwner, agentId } = input;

  if (senderId && messageChannel) {
    for (const owner of normalizedOwners) {
      if ((owner.channels ?? {})[messageChannel] === senderId) {
        return {
          ownerId: owner.owner_id,
          ownerNamespace: owner.owner_namespace,
        };
      }
    }
  }

  if (senderId) {
    for (const owner of normalizedOwners) {
      if (Object.values(owner.channels ?? {}).includes(senderId)) {
        return {
          ownerId: owner.owner_id,
          ownerNamespace: owner.owner_namespace,
        };
      }
    }
  }

  if (senderIsOwner && normalizedOwners.length === 1) {
    const owner = normalizedOwners[0];
    return {
      ownerId: owner.owner_id,
      ownerNamespace: owner.owner_namespace,
    };
  }

  if (normalizedOwners.length > 0) {
    const owner = normalizedOwners[0];
    return {
      ownerId: owner.owner_id,
      ownerNamespace: owner.owner_namespace,
    };
  }

  if (senderId) {
    return {
      ownerId: senderId,
      ownerNamespace: "default",
    };
  }

  if (agentId) {
    return {
      ownerId: agentId,
      ownerNamespace: "default",
    };
  }

  return null;
}

export function isAgentWhitelisted(agentId: string | undefined, whitelist?: string[]): boolean {
  if (!agentId) return false;
  return (whitelist ?? []).includes(agentId);
}

export function canAccessScope(
  agentId: string | undefined,
  scope: MemoryScope,
  whitelist?: string[],
): boolean {
  if (!agentId) return false;
  if (agentId === "tiffany-customer" && scope === "owner_shared") return false;

  switch (scope) {
    case "owner_shared":
      return isAgentWhitelisted(agentId, whitelist);
    case "agent_local":
    case "session_distilled":
      return true;
    default:
      return false;
  }
}

export function validateAccess(
  agentId: string | undefined,
  targetScope: MemoryScope,
  whitelist?: string[],
): AccessResult {
  if (canAccessScope(agentId, targetScope, whitelist)) {
    return { allowed: true };
  }

  if (agentId === "tiffany-customer" && targetScope === "owner_shared") {
    return {
      allowed: false,
      reason: "tiffany-customer hard-deny for owner_shared",
    };
  }

  if (targetScope === "owner_shared") {
    return {
      allowed: false,
      reason: `agent ${agentId ?? "unknown"} is not whitelisted for owner_shared`,
    };
  }

  return {
    allowed: false,
    reason: `agent ${agentId ?? "unknown"} cannot access scope ${targetScope}`,
  };
}

export function getAccessibleScopes(agentId: string | undefined, whitelist?: string[]): MemoryScope[] {
  const scopes: MemoryScope[] = [];
  for (const scope of ["agent_local", "session_distilled", "owner_shared"] as const) {
    if (canAccessScope(agentId, scope, whitelist)) scopes.push(scope);
  }
  return scopes;
}
