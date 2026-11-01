// Preset contract dump: the REAL frontend presets.ts as v4 declarations,
// replicating the editor's serializeGraph (typed params off the registry
// ptype, frame contract with feedback CTRL edges). Node-side only: the
// canvas runtime never runs. Usage: node _presets_dump.mjs <bundle.mjs>
import { readFileSync } from 'node:fs';

const bundle = await import(process.argv[2]);
const PRESETS = bundle.PRESETS;
const registry = JSON.parse(readFileSync(new URL(
  '../../../../web-ui-v5/webui/src/pages/workflow/registry.json', import.meta.url), 'utf-8'));
const components = registry.components;

const paramTypeOf = (nodeType, name) =>
  (components[nodeType]?.params ?? []).find((p) => p.name === name)?.type;

const coerceParam = (value, ptype) => {
  if (!ptype || ptype === 'str' || ptype === 'code') return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};

const portKind = (nodeType, portKey, direction) =>
  (components[nodeType]?.[direction === 'in' ? 'ports_in' : 'ports_out'] ?? [])
    .find((p) => p.key === portKey)?.kind;

const defaultToText = (value) => {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const declarations = PRESETS.map((preset) => {
  const nodes = preset.nodes.map((n) => ({
    id: n.id,
    type: n.type,
    params: Object.fromEntries(
      (n.params ?? components[n.type].params.map((p) => ({ label: p.name, value: defaultToText(p.default) })))
        .map((p) => [p.label, coerceParam(p.value, paramTypeOf(n.type, p.label))]),
    ),
  }));
  const edges = preset.edges.map((e) => ({
    id: `${e.from}.${e.fromPort}-${e.to}.${e.toPort}${e.slot ? `#${e.slot}` : ''}`,
    source: e.from,
    source_port: e.fromPort,
    target: e.to,
    target_port: e.toPort,
    ...(e.slot ? { slot: e.slot } : {}),
  }));
  const typeOf = (id) => preset.nodes.find((n) => n.id === id)?.type ?? '';
  const frames = (preset.frames ?? []).map((f) => {
    const members = f.memberIds ?? [];
    const sources = members.filter((id) => components[typeOf(id)]?.lifecycle === 'Iterator');
    const consumers = members.filter((id) => components[typeOf(id)]?.lifecycle !== 'Iterator');
    const owner = consumers.find((id) => (components[typeOf(id)]?.effects ?? []).includes('train'));
    return {
      id: f.id,
      name: f.label ?? '',
      members,
      sources,
      consumers,
      budget_owner: owner ?? consumers[0] ?? '',
      feedback: edges
        .filter((e) => members.includes(e.source) && sources.includes(e.target))
        .filter((e) => portKind(typeOf(e.target), e.target_port, 'in') === 'CTRL')
        .map((e) => e.id),
    };
  });
  return { version: 4, workflow_id: preset.key, nodes, edges, frames };
});

process.stdout.write(JSON.stringify(declarations));
