import importlib
import os
import pkgutil

# Auto-discover and import all provider modules
package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f"llm_council.providers.{module_name}")
