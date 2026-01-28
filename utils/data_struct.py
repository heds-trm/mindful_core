import itertools
from typing import Hashable, Mapping, TypeVar, Iterable

_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT")


class AliasDict(Mapping[_KT, _VT]):
    def __init__(self, **kwargs):
        self._aliases: dict[_KT, _KT] = {}
        self._values: dict[_KT, _VT] = {}
        for key, value in kwargs.items():
            self[key] = value

    def __getitem__(self, key: _KT) -> _VT:
        if key in self._values:
            return self._values[key]

        if key in self._aliases:
            return self._values[self._aliases[key]]

        raise KeyError(key)

    def __setitem__(self, key: _KT | tuple[_KT, ...], value: _VT) -> None:
        if isinstance(key, tuple):
            if len(key) == 0:
                raise ValueError("Key tuple is empty.")
            key, *aliases = key
            self._add_aliases(key, *aliases)

        elif key in self._aliases:
            key = self._aliases[key]

        self._values[key] = value

    def _add_aliases(self, key: _KT, *aliases: _KT) -> None:
        if key in self._aliases:
            key = self._aliases[key]

        for alias in aliases:
            if alias in self._values:
                if alias == key:
                    continue
                self._values.pop(alias)
                for existing_alias in self._aliases.keys():
                    if self._aliases[existing_alias] == alias:
                        self._aliases[existing_alias] = key
            self._aliases[alias] = key

    def add_aliases(self, key: _KT, *aliases: _KT) -> None:
        if (key not in self._values) and (key not in self._aliases):
            raise KeyError("Unknown original key to point to `{}`".format(key))

        self._add_aliases(key, *aliases)

    def __contains__(self, key: _KT) -> bool:
        return (key in self._values) or (key in self._aliases)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterable[_KT]:
        return iter(self._values)

    def keys(self) -> Iterable[_KT]:
        return self._values.keys()

    def aliases(self) -> Iterable[_KT]:
        return self._aliases.keys()

    def keys_and_aliases(self) -> Iterable[_KT]:
        return itertools.chain(self.keys(), self.aliases())

    def values(self) -> Iterable[_VT]:
        return self._values.values()

    def items(self) -> Iterable[tuple[_KT, _VT]]:
        return self._values.items()

    def keys_to_aliases(self) -> dict[_KT, list[_KT]]:
        result = {}
        for alias, original_key in self._aliases.items():
            if original_key not in result:
                result[original_key] = []
            result[original_key].append(alias)

        return result

    def keys_to_aliases_repr(self) -> list[str]:
        keys_to_aliases = self.keys_to_aliases()

        key_aliases_representations = []
        for key in self._values:
            if (key in keys_to_aliases) and (len(keys_to_aliases[key]) > 0):
                aliases_rep = ", ".join(keys_to_aliases[key])
                key_repr = "{} (aliases: {})".format(key, aliases_rep)
            else:
                key_repr = key
            key_aliases_representations.append(key_repr)

        return key_aliases_representations

    def __repr__(self) -> str:
        key_aliases_representations = self.keys_to_aliases_repr()
        kv_representations = []
        for key_repr, value in zip(key_aliases_representations, self._values.values()):
            kv_repr = "{}: {}".format(key_repr, value)
            kv_representations.append(kv_repr)
        result = "{" + ", ".join(kv_representations) + "}"
        return result
