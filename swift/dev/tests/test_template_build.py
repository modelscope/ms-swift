"""What class does build_template hand back, and what does that class still know how to do?

The default path must ADD dev's next-token label shift to the legacy template without REPLACING its
class. That distinction is the whole content of these tests: `copy.copy` + `__class__ = DevSubclass`
(what build_template used to do unconditionally) drops every method the legacy family overrode --
measured on Qwen3.5: 14, including `_encode`, `replace_tag`, `_data_collator`, `_get_position_ids`,
`packing_row` -- and makes `super()._encode()` resolve to the BASE legacy `_encode` instead of the
family's. Text-only models never noticed, because for `qwen2_5`/`qwen3` the legacy class IS the base;
multimodal families lose their media handling entirely, which is what "dev VL never calls
fetch_image" turned out to be.
"""
import pickle

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def test_default_path_shifts_labels_without_replacing_the_legacy_class():
    """End of the real pipeline: legacy's own class stays in the MRO, and labels come out shifted."""
    from swift.dev.builders import build_template
    from swift.dev.configs import TemplateConfig
    from swift.dev.template import DevMixin
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.base import Template as LegacyBase

    _, proc = get_model_processor(MODEL, load_model=False)
    cfg = TemplateConfig(template='qwen2_5', max_length=256)
    tpl = build_template(cfg, proc)

    legacy_cls = type(get_template(proc, template_type='qwen2_5', max_length=256))
    mro = type(tpl).__mro__
    assert legacy_cls in mro, f'legacy class {legacy_cls.__name__} was replaced, not extended'
    assert DevMixin in mro

    msgs = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello there'}]
    encoded = tpl.encode({'messages': msgs})
    legacy_tpl = get_template(proc, template_type='qwen2_5', max_length=256)
    legacy_tpl.set_mode('train')
    legacy_encoded = legacy_tpl.encode({'messages': msgs})

    assert list(encoded['input_ids']) == list(legacy_encoded['input_ids'])
    # dev's whole delta: legacy's labels, moved one position left (contract 1).
    assert list(encoded['labels']) == list(legacy_encoded['labels'])[1:] + [-100]
    assert encoded[DevMixin.SHIFTED_KEY] is True


def test_derived_class_keeps_family_methods_and_survives_pickling():
    """Structural guarantees, on a stand-in family so no VL model has to be downloaded.

    Pickling is not hypothetical: build_dataset hands the template to
    EncodePreprocessor/AddLengthPreprocessor/PackingDataset, and datasets.map pickles those whenever
    dataset_num_proc > 1. pickle resolves a class by module + qualname, so a `type(...)` result that
    is not reachable as a module attribute fails there and nowhere else.
    """
    from swift.dev.template import DevMixin, shifted_template_class

    class _Family:
        """Stands in for e.g. Qwen3_5Template: a legacy subclass with its own overrides."""

        is_training = True

        def encode(self, inputs, return_template_inputs=False, return_length=False):
            return {'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}

        def replace_tag(self, *args):
            return ['family-placeholder']

    cls = shifted_template_class(_Family)
    assert cls.__mro__[1:3] == (DevMixin, _Family)
    assert cls.replace_tag is _Family.replace_tag, 'family override stopped dispatching'
    # Cached per base: a fresh class per call would make isinstance across two templates of the same
    # family false and multiply the pickle registrations.
    assert shifted_template_class(_Family) is cls

    obj = cls()
    assert obj.encode({})['labels'] == [2, 3, -100]
    assert obj.replace_tag() == ['family-placeholder']
    # _Family is local to this test so the instance cannot be pickled; the CLASS being resolvable is
    # what pickle needs from us, and it is the part that a plain type() call breaks.
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_shifted_class_unpickles_in_a_process_that_never_created_it():
    """The pickle blob must load in a FRESH interpreter, not just in the process that built the class.

    This is the case datasets.map actually exercises: its workers are new processes that merely import
    swift.dev.template.template, so `globals()[name] = cls` -- which only populates the creating
    process -- left the name unresolvable there. The failure was
    `AttributeError: Can't get attribute 'ShiftedTemplate'`, and it appeared ONLY under
    dataset_num_proc > 1, i.e. never in a single-process test. The module-level __getattr__ rebuilds
    the class on lookup instead.

    A real subprocess is used rather than a mock of the import machinery, because what broke was the
    interaction between pickle's by-name resolution and per-process module state -- the one thing an
    in-process test cannot see.
    """
    import subprocess
    import sys

    from swift.dev.template import shifted_template_class
    from swift.template.base import Template as LegacyTemplate

    blob = pickle.dumps(shifted_template_class(LegacyTemplate))
    child = subprocess.run([
        sys.executable, '-c', 'import pickle, sys, swift.dev.template.template;'
        'cls = pickle.loads(sys.stdin.buffer.read());'
        'print(cls.__module__ + "." + cls.__qualname__)'
    ],
                           input=blob,
                           capture_output=True,
                           timeout=600)
    assert child.returncode == 0, (
        f'a fresh process could not unpickle the shifted class:\n{child.stderr.decode()[-800:]}')
    assert child.stdout.decode().strip() == 'swift.dev.template.template.ShiftedTemplate', \
        child.stdout.decode()


def test_module_getattr_rejects_unrelated_names():
    """__getattr__ must not turn every missing attribute into a class.

    A module __getattr__ that answers anything would mask genuine typos and break `hasattr` probes
    elsewhere, so only the Shifted* namespace is served -- and only when a matching legacy class
    exists.
    """
    import swift.dev.template.template as mod

    for name in ('not_a_class', 'ShiftedNoSuchTemplateFamily'):
        try:
            getattr(mod, name)
        except AttributeError:
            continue
        raise AssertionError(f'__getattr__ answered {name!r} instead of raising AttributeError')
