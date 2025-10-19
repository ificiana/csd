from pathlib import Path


def find_mmap_files(root: str, recursive: bool = True):
    rootp = Path(root)
    if recursive:
        return sorted(p.resolve() for p in rootp.rglob("*.mmap"))
    else:
        return sorted(p.resolve() for p in rootp.glob("*.mmap"))


tpl1 = r"""
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b133669",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "sys.path.append(str(Path(\"..\").resolve()))\n",
    "\n",
    "from tel import MMapJSON\n",
    "from viz.utils import Data, plot_graphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1447805",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = Data.model_validate(MMapJSON(\"../data/"""
tpl2 = r"""\", 2**24).read())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "329c2073",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_graphs(data)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""

files = find_mmap_files("data", recursive=True)
for f in files:
    tpl = tpl1 + f.name + tpl2
    ts = f.name.replace(".mmap", "")
    with open(f"viz/{ts}.ipynb", "w") as f:
        f.write(tpl)
