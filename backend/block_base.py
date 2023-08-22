import json
import uuid
from typing import Dict, List, Union, Any, Optional, Type

from main import singleton as gs

plugin_info = {"name": "Blocks v2"}

# Global list to store available blocks
AVAILABLE_BLOCKS = []

def register_class(cls: Type) -> Type:
    """Decorator to register a class in the global list of available blocks.

    Args:
        cls (Type): The class to be registered.

    Returns:
        Type: The registered class.
    """
    if cls.__name__ not in [existing_cls.__name__ for existing_cls in AVAILABLE_BLOCKS]:
        AVAILABLE_BLOCKS.append(cls)
    return cls

class BaseWidget:
    """Base class for all widget types."""

    name: str = "Widget"
    widget_type: str = "text"  # default type
    multiline: bool = False
    precision: str = "int"
    value: Union[str, int, float, bool, None] = None
    options: List[Union[str, int, float]] = []  # This will store the available options for widgets like selectbox
    selected_index: int = 0  # This will store the selected index for widgets like selectbox

    def __init__(self) -> None:
        """Initializes a new BaseWidget instance."""
        if self.widget_type == "text":
            self.widget_type = self.widget_type if not self.multiline else "text_multi"
        self.uid: str = str(uuid.uuid4())  # Unique ID for each widget instance

    def get(self) -> Any:
        """Gets the widget.

        Returns:
            Any: The widget object.
        """
        return self.widget

    def get_value(self) -> Any:
        """Gets the value of the widget.

        Returns:
            Any: The value of the widget.
        """
        if hasattr(self.widget, 'value'):
            return self.widget.value()

    def serialize(self) -> Dict[str, Union[str, bool, Union[str, int, float, bool], List[Union[str, int, float]], int]]:
        """Serializes the widget into a dictionary.

        Returns:
            Dict[str, Union[str, bool, Union[str, int, float, bool], List[Union[str, int, float]], int]]: The serialized widget data.
        """
        return {
            'uid': self.uid,
            'name': self.name,
            'widget_type': self.widget_type,
            'multiline': self.multiline,
            'precision': self.precision,
            'value': self.value,
            'options': self.options,
            'selected_index': self.selected_index
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Union[str, bool, Union[str, int, float, bool], List[Union[str, int, float]], int]]) -> 'BaseWidget':
        """Deserializes the widget from a dictionary.

        Args:
            data (Dict[str, Union[str, bool, Union[str, int, float, bool], List[Union[str, int, float]], int]]): The serialized widget data.

        Returns:
            BaseWidget: The deserialized widget object.
        """
        widget = cls()
        widget.uid = data['uid']
        widget.name = data['name']
        widget.widget_type = data['widget_type']
        widget.multiline = data['multiline']
        widget.precision = data['precision']
        widget.value = data['value']
        widget.options = data['options']
        widget.selected_index = data['selected_index']
        return widget

class BaseBlock:
    """Base class for all block types."""

    widgets: List[BaseWidget]

    def __init__(self) -> None:
        """Initializes a new BaseBlock instance."""
        self.uid: str = str(uuid.uuid4())  # Unique ID for each widget instance
        self.widgets = []

    def serialize(self) -> Dict[str, Any]:
        """Serializes the block into a dictionary.

        Returns:
            Dict[str, Any]: The serialized block data.
        """
        return {
            'type': self.__class__.__name__,
            'widgets': [widget.serialize() for widget in self.widgets]
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'BaseBlock':
        """Deserializes the block from a dictionary.

        Args:
            data (Dict[str, Any]): The serialized block data.

        Returns:
            BaseBlock: The deserialized block object.
        """
        block_type_name = data['type']
        for available_block in AVAILABLE_BLOCKS:
            if available_block.__name__ == block_type_name:
                block_class = available_block
                break
        else:
            # If we don't find the correct type, default to BaseBlock
            block_class = BaseBlock

        block = block_class()
        block.widgets = [BaseWidget.deserialize(widget_data) for widget_data in data['widgets']]
        return block

    def text(self,
             label: str = 'default_label',
             value: str = 'default_value',
             multiline: bool = False,
             expose: bool = True) -> None:
        """Adds a text widget to the block.

        Args:
            label (str, optional): The label of the widget. Defaults to 'default_label'.
            value (str, optional): The default value of the widget. Defaults to 'default_value'.
            multiline (bool, optional): Specifies if the text widget is multiline. Defaults to False.
            expose (bool, optional): Specifies if the widget is exposed in the UI. Defaults to True.
        """
        text_widget = BaseWidget()
        text_widget.widget_type = "text" if not multiline else "text_multi"
        text_widget.value = value
        text_widget.name = label
        text_widget.multiline = multiline
        text_widget.expose = expose
        self.widgets.append(text_widget)

    def number(self,
               label: str = 'default_label',
               value: Union[int, float] = 1.0,
               step: Union[int, float] = 1,
               min_val: Union[int, float] = 1,
               max_val: Union[int, float] = 10,
               type_val: Optional[str] = 'entry',
               expose: bool = True) -> None:
        """Adds a number widget to the block.

        Args:
            label (str, optional): The label of the widget. Defaults to 'default_label'.
            value (Union[int, float], optional): The default value of the widget. Defaults to 1.0.
            step (Union[int, float], optional): The step value for the widget. Defaults to 1.
            min_val (Union[int, float], optional): The minimum value for the widget. Defaults to 1.
            max_val (Union[int, float], optional): The maximum value for the widget. Defaults to 10.
            type_val (Optional[str], optional): The type of the widget ('entry' or 'slider'). Defaults to 'entry'.
            expose (bool, optional): Specifies if the widget is exposed in the UI. Defaults to True.
        """
        number_widget = BaseWidget()
        number_widget.name = label
        number_widget.widget_type = "number" if type_val == 'entry' else 'slider'
        number_widget.value = value
        number_widget.step = step
        number_widget.min = min_val
        number_widget.max = max_val
        number_widget.expose = expose
        self.widgets.append(number_widget)

    def dropdown(self,
                  label: str = 'default_label',
                  options: List[str] = ["default"],
                  index: int = 0,
                  expose: bool = True) -> None:
        """Adds a dropdown widget to the block.

        Args:
            label (str, optional): The label of the widget. Defaults to 'default_label'.
            options (List[str], optional): The available options for the dropdown. Defaults to ["default"].
            index (int, optional): The default selected index. Defaults to 0.
            expose (bool, optional): Specifies if the widget is exposed in the UI. Defaults to True.
        """
        selectbox = BaseWidget()
        selectbox.widget_type = "dropdown"
        selectbox.options = options
        selectbox.index = index
        selectbox.name = label
        selectbox.expose = expose
        self.widgets.append(selectbox)

    def checkbox(self,
                 label: str = 'default_label',
                 value: bool = False,
                 expose: bool = True) -> None:
        """Adds a checkbox widget to the block.

        Args:
            label (str, optional): The label of the widget. Defaults to 'default_label'.
            value (bool, optional): The default value of the widget (checked or not). Defaults to False.
            expose (bool, optional): Specifies if the widget is exposed in the UI. Defaults to True.
        """
        checkbox_widget = BaseWidget()
        checkbox_widget.widget_type = "checkbox"
        checkbox_widget.value = value
        checkbox_widget.name = label
        checkbox_widget.expose = expose
        self.widgets.append(checkbox_widget)

    def multiselect(self,
                    label: str = 'default_label',
                    options: List[str] = ["default"],
                    default: List[str] = [],
                    expose: bool = True) -> None:
        """Adds a multiselect widget to the block.

        Args:
            label (str, optional): The label of the widget. Defaults to 'default_label'.
            options (List[str], optional): The available options for the multiselect. Defaults to ["default"].
            default (List[str], optional): The default selected options. Defaults to an empty list.
            expose (bool, optional): Specifies if the widget is exposed in the UI. Defaults to True.
        """
        multiselect_widget = BaseWidget()
        multiselect_widget.widget_type = "multiselect"
        multiselect_widget.options = options
        multiselect_widget.value = default
        multiselect_widget.name = label
        multiselect_widget.expose = expose
        self.widgets.append(multiselect_widget)

class BlockHolder:
    """Holder class for blocks."""

    blocks: List[BaseBlock]

    def serialize(self) -> Dict[str, List[Dict[str, Any]]]:
        """Serializes the block holder into a dictionary.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The serialized block holder data.
        """
        return {
            'blocks': [block.serialize() for block in self.blocks]
        }

    @classmethod
    def deserialize(cls, data: Dict[str, List[Dict[str, Any]]]) -> 'BlockHolder':
        """Deserializes the block holder from a dictionary.

        Args:
            data (Dict[str, List[Dict[str, Any]]]): The serialized block holder data.

        Returns:
            BlockHolder: The deserialized block holder object.
        """
        holder = cls()
        holder.blocks = [BaseBlock.deserialize(block_data) for block_data in data['blocks']]
        return holder

def save_to_json(filename: str) -> None:
    """Saves the blocks data to a JSON file.

    Args:
        filename (str): The path to the file where the data will be saved.
    """
    serialized_blocks = [block.serialize() for block in gs.data['added_blocks']]
    with open(filename, 'w') as file:
        json.dump({"blocks": serialized_blocks}, file)

def load_from_json(filename: str) -> None:
    """Loads the blocks data from a JSON file.

    Args:
        filename (str): The path to the file from where the data will be loaded.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    gs.data['added_blocks'] = [BaseBlock.deserialize(block_data) for block_data in data['blocks']]
