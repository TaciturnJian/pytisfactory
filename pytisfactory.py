# 使用中文注释

from dataclasses import dataclass
from enum import Enum
import re
import logging

#region Code Guard

class CodeGuard:
    @staticmethod
    def TypeAssert(obj, expect, message: str|None = None):
        if not isinstance(obj, expect):
            if message == None:
                message = "类型检查失败：期望为{}，实际为{}".format(expect, type(obj))
            logging.error(message)
            raise TypeError(message)

#endregion

#region Game Elements

ALLOW_PARALLEL_GROUP_SUBGRAPH = True
ALLOW_SERIAL_GROUP_SUBGRAPH = True
ALLOW_MERMAID_COMMENT = False

def FloatStr(v: float) -> str:
    # 优先整数
    if v.is_integer():
        return str(int(v))
    return str(v)

@dataclass
class Item:
    """
    表示一个具有名称、数量和后缀单位的物品。
    字符串格式为 "${name}(${count}${postfix})"。
    """
    name: str = "物品"
    count: float = 0.0
    postfix: str = "单位"
    
    """
    比较当前 Item 实例与另一个 Item 或字符串是否相等。
    如果 'other' 是字符串，则通过 parse 将其转换为 Item 实例。
    只有当 name 和 count 属性都相等时返回 True，否则返回 False。
    参数:
        other (Item | str): 要比较的 Item 实例或字符串。
    返回:
        bool: 如果两个物品相等返回 True，否则返回 False。
    """
    def equal(self, other: 'Item|str') -> bool:
        if isinstance(other, str):
            temp = Item()
            if not temp.parse(other):
                return False
            other = temp
        
        return self.name == other.name and self.count == other.count
    
    def name_equal(self, name: 'Item|str') -> bool:
        if isinstance(name, Item):
            name = name.name
        return self.name == name
    
    def simplify(self) -> 'Item':
        self.name = self.name.strip()
        self.postfix = self.postfix.strip()
        return self
    
    def parse(self, text: str) -> bool:
        self.name = ""
        self.count = 0.0
        self.postfix = ""
        text = text.strip()
        if not text:
            return True
        try:
            match = re.match(r'^(.+)\((\d+\.?\d*)(.*)\)$', text.strip())
            
            if match:
                self.name = match.group(1)
                self.count = float(match.group(2))
                self.postfix = match.group(3)
                self.simplify()
                logging.info("从字符串({})中提取物品数据({})".format(text, self))
                return True
            else:
                return False
        except Exception as e:
            logging.warning("无法从字符串({})中提取物品数据：{}".format(text, e))
            return False
    
    def copy(self) -> 'Item':
        return Item(self.name, self.count, self.postfix)

    def multiply(self, factor: float) -> 'Item':
        self.count *= factor
        return self

    def empty(self) -> bool:
        return self.count <= 0 or len(self.name) == 0
    
    def __str__(self) -> str:
        return "{}({}{})".format(self.name, FloatStr(self.count), self.postfix)

class Power(Item):
    def __init__(self, count: float = 0):
        super().__init__("电力", count, "MW")

class ItemList:
    def __init__(self, items: list[Item]|None = None):
        self.items = [] if items is None else items
        
    def equal(self, other: 'ItemList') -> bool:
        my_len = len(self.items)
        if my_len != len(other.items):
            return False
        
        for i in range(my_len):
            if not self.items[i].equal(other.items[i]):
                return False
        return True
    
    def has_item(self, item: Item|str) -> bool:
        if isinstance(item, Item):
            item = item.name
        for my_item in self.items:
            if my_item.name_equal(item):
                return True
        return False
    
    def index_of(self, item: Item|str) -> int:
        if isinstance(item, Item):
            item = item.name
        for i, my_item in enumerate(self.items):
            if my_item.name_equal(item):
                return i
        return -1
    
    def at(self, index: int) -> Item | None:
        if index < 0 or index >= len(self.items):
            logging.warning("尝试访问 ItemList 中不存在的索引({})".format(index))
            return None
        return self.items[index]

    def simplify(self) -> 'ItemList':
        # 先简化所有物品，然后进行排序
        for item in self.items:
            item.simplify()
        self.items.sort(key=lambda item: item.name)
        
        simplified_items = list()
        for item in self.items:
            if not simplified_items or not simplified_items[-1].equal(item):
                simplified_items.append(item.copy())
            else:
                simplified_items[-1].count += item.count
        
        self.items = [item for item in simplified_items if not item.empty()]
        return self
        
    def append(self, item: Item|str) -> 'ItemList':
        if isinstance(item, str):
            item = Item(item)
            
        item.simplify()
        if item.empty():
            logging.warning("尝试添加空物品({})到物品列表".format(item))
            return self
        
        self.items.append(item)
        return self
    
    def add(self, item: Item|str) -> 'ItemList':
        if isinstance(item, str):
            item = Item(item)
        item.simplify()
        if item.empty():
            logging.warning("尝试添加空物品({})到物品列表".format(item))
            return self
        logging.info("向物品列表添加物品：{}".format(item))
        index = self.index_of(item)
        if index == -1:
            self.items.append(item)
        else:
            self.items[index].count += item.count
        return self
    
    def merge(self, items: 'ItemList|list[Item]') -> 'ItemList':
        if isinstance(items, ItemList):
            items = items.items

        for item in items:
            self.add(item)
            
        return self.simplify()

    def pop(self, item: Item|str) -> Item | None:
        index = self.index_of(item)
        if index != -1:
            return self.items.pop(index)
        return None

    def parse(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        self.items = list()
        new_item = Item()
        
        failed_count = 0
        succeeded_count = 0
        for word in text.split(","):
            if new_item.parse(word.strip()):
                self.items.append(new_item.copy())
                new_item = Item()
                succeeded_count += 1
            else:
                failed_count += 1
        logging.info("从字符串({})中解析物品列表，成功{}个，失败{}个".format(text, succeeded_count, failed_count))
        return succeeded_count > 0

    def copy(self) -> 'ItemList':
        return ItemList(
            [item.copy() for item in self.items]
        )
    
    def clear(self):
        self.items = list()
      
    def multiply(self, factor: float) -> 'ItemList':
        for item in self.items:
            item.multiply(factor)
        return self

    def empty(self) -> bool:
        return not self.items

    def power(self) -> Power:
        power = Power()
        index = self.index_of(power)
        if index < 0:
            return power
        return Power(self.items[index].count)
    
    def pop_power(self) -> Power:
        power = self.pop(Power())
        if power == None:
            return Power()
        return Power(power.count)

    def minus(self, items: 'ItemList|list[Item]') -> 'ItemList':
        if isinstance(items, ItemList):
            items = items.items
            
        self.simplify()
        for item in items:
            index = self.index_of(item)
            if index < 0:
                raise ValueError("尝试从物品列表中减去不存在的物品：{}".format(item))
            my_item = self.items[index]
            if my_item.count < item.count:
                raise ValueError("尝试从物品列表中减去数量过多的物品：{}，当前只有{}".format(item, my_item))
            my_item.count -= item.count
        
        return self.simplify()

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index: int) -> Item | None:
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def __str__(self) -> str:
        return  ','.join(str(item) for item in self.items)

@dataclass
class Machine:
    name: str = "机器"                  # 机器名称
    cost: ItemList = ItemList()     # 建造成本，一次性
    inputs: ItemList = ItemList()   # 输入物品，每分钟消耗
    outputs: ItemList = ItemList()  # 输出物品，每分钟产生

    def equal(self, other: 'Machine') -> bool:
        return (self.name == other.name and
                self.cost.equal(other.cost) and
                self.inputs.equal(other.inputs) and
                self.outputs.equal(other.outputs))

    def simplify(self) -> 'Machine':
        self.name = self.name.strip()
        self.cost.simplify()
        self.inputs.simplify()
        self.outputs.simplify()
        return self
        
    def architect(self) -> str:
        return self.name

    def parse(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        try:
            match = re.match(r'^(.+)\[(.*)\]\-\((.*)\)\+\((.*)\)$', text)
            if match:
                self.name = match.group(1)
                cost_text = match.group(2)
                input_text = match.group(3)
                output_text = match.group(4)
                
                if not self.cost.parse(cost_text):
                    logging.warning("无法从字符串({})中解析机器的建造成本：{}".format(text, cost_text))
                if not self.inputs.parse(input_text):
                    logging.warning("无法从字符串({})中解析机器的输入物品：{}".format(text, input_text))
                if not self.outputs.parse(output_text):
                    logging.warning("无法从字符串({})中解析机器的输出物品：{}".format(text, output_text))
                return True
            return False

        except Exception as e:
            logging.warning("无法从字符串({})中解析机器数据：{}".format(text, e))
            return False
    
    def copy(self) -> 'Machine':
        return Machine(
            name=self.name,
            cost=self.cost.copy(),
            inputs=self.inputs.copy(),
            outputs=self.outputs.copy()
        )

    def clear(self) -> 'Machine':
        self.name = ""
        self.cost.clear()
        self.inputs.clear()
        self.outputs.clear()
        return self

    def empty(self) -> bool:
        return not self.name
    
    def __str__(self) -> str:
        return "{}[{}]-({})+({})".format(
            self.name, 
            self.cost, 
            self.inputs, 
            self.outputs)

class Buffer(Machine):
    def __init__(self, item: Item, name: str|None = None):
        if name is None:
            name = "{}缓冲".format(item.name)
        super().__init__(
            name, 
            ItemList(),
            ItemList([item]),
            ItemList([item]))

class Composition(Enum):
    unknown = "未知"
    serial = "串联"
    parallel = "并联"

class NameIterator:
    def __init__(self, begin: str = 'aaa'): # 至少要三个字符才保证 mermaid 图正常
        for char in begin:
            if char < 'a' or char > 'z':
                begin = 'a'
                logging.warning("NameIterator 的起始字符({})不在 a-z 范围内，已重置为 '{}'".format(char, begin))
                break
        self._current = begin
        logging.info("NameIterator 初始化，起始名称为：{}".format(self.current))

    @property
    def current(self):
        return self._current
    
    def peek(self):
        """
        返回下一个名称，但不会推进当前状态。
        """
        return self._increment(self._current)

    def step(self):
        """
        将迭代器推进到下一个值，更新内部状态。

        Returns:
            str: 返回更新后的值。
        """
        self._current = self._increment(self._current)
        return self._current
    
    @staticmethod
    def _increment(s):
        chars = list(s)
        i = len(chars) - 1
        
        while i >= 0:
            if chars[i] < 'z':
                chars[i] = chr(ord(chars[i]) + 1)
                return ''.join(chars)
            else:
                chars[i] = 'a'
                i -= 1
        
        return 'a' * (len(s) + 1)

def mermaid_element(name: str, label: str = "") -> str:
    """
    生成 mermaid 图节点字符串，格式为 name["label"]。
    节点是 mermaid 中的基本元素，互相连接构成 mermaid 图。

    Args:
        name (str): 节点名称。
        label (str, optional): 节点标签。默认为空字符串。

    Returns:
        str: 格式化后的 mermaid 节点字符串。
    """
    name = name.strip()
    label = label.strip()
    if label != "":
        label = '["{}"]'.format(label)
    return "{}{}".format(name, label)

def mermaid_link(from_id: str, to_id: str, comment: str|ItemList = "") -> str:
    """
    生成 mermaid 图的连接字符串，格式为 from_id-->|"comment"|to_id。
    用于在 mermaid 图中连接两个节点，并可选添加注释。

    Args:
        from_id (str): 起始节点的名称。
        to_id (str): 目标节点的名称。
        comment (str | ItemList, optional): 连接上的注释内容，可以为字符串或 ItemList。默认为空字符串，也成为 label。

    Returns:
        str: 格式化后的 mermaid 连接字符串。
    """
    if isinstance(comment, ItemList):
        comment = str(comment)
    comment = comment.strip()
    if comment != "":
        comment = '|"{}"|'.format(comment)
    return "{}-->{}{}".format(from_id, comment, to_id)

class MachineList(Machine):
    def __init__(self, name: str = "机器列表", cost: ItemList = ItemList(), inputs: ItemList = ItemList(), outputs: ItemList = ItemList(), machines: list[Machine] = list(), composition: str = Composition.unknown.value):
        if not machines:
            machines = list()
        super().__init__(name=name, cost=cost, inputs=inputs, outputs=outputs)
        self.machines = machines
        self.composition = composition
    
    def _parse_machines(self, text: str) -> bool:
        self.machines = list()
        try:
            machine_texts = text.split(';')
            failed_count = 0
            succeeded_count = 0
            for machine_text in machine_texts:
                machine = Machine()
                if not machine.parse(machine_text):
                    logging.warning("无法从字符串({})中解析机器：{}".format(text, machine_text))
                    failed_count += 1
                else:
                    succeeded_count += 1
                    self.machines.append(machine)
            return succeeded_count > 0
        except Exception as e:
            logging.error("无法从字符串({})中解析机器列表：{}".format(text, e))
            return False

    def copy(self) -> 'MachineList':
        return MachineList(
            self.name,
            self.cost.copy(),
            self.inputs.copy(),
            self.outputs.copy(),
            [machine.copy() for machine in self.machines],
            self.composition
        )
        
    def reinit(self) -> 'MachineList':
        if not self.machines:
            return self
        
        self.cost.clear()
        self.inputs.clear()
        self.outputs.clear()
        
        if self.composition == Composition.parallel.value:
            for machine in self.machines:
                self.cost.merge(machine.cost)
                self.inputs.merge(machine.inputs)
                self.outputs.merge(machine.outputs)
            return self
        
        if self.composition == Composition.serial.value:
            self.cost = self.machines[0].cost.copy()
            self.inputs = self.machines[0].inputs.copy()
            self.outputs = self.machines[0].outputs.copy()
            self.outputs.pop_power()
            for machine in self.machines[1:]:
                self.cost.merge(machine.cost)
                consumption = machine.inputs.copy()
                consumption.pop_power()
                self.outputs.minus(consumption)
                self.outputs.merge(machine.outputs)
            return self

        raise ValueError("未知的组合类型")

    def architect(self) -> str:
        if self.composition == Composition.parallel.value:
            return "{}({})".format(self.name, '||'.join(machine.architect() for machine in self.machines))
        if self.composition == Composition.serial.value:
            return "{}({})".format(self.name, '->'.join(machine.architect() for machine in self.machines))
        raise ValueError("未知的组合类型")

    def parse(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        try:
            match = re.match(r'^(.+)\[(.*)\]\-\((.*)\)\+\((.*)\)<(.*)>\((.*)\)$', text)
            if match:
                self.name = match.group(1)
                cost_text = match.group(2)
                input_text = match.group(3)
                output_text = match.group(4)
                self.composition = match.group(5)
                machine_text = match.group(6)
                
                if not self.cost.parse(cost_text):
                    logging.warning("无法从字符串({})中解析机器的建造成本：{}".format(text, cost_text))
                if not self.cost.parse(cost_text):
                    logging.warning("无法从字符串({})中解析机器的建造成本：{}".format(text, cost_text))
                if not self.inputs.parse(input_text):
                    logging.warning("无法从字符串({})中解析机器的输入物品：{}".format(text, input_text))
                if not self.outputs.parse(output_text):
                    logging.warning("无法从字符串({})中解析机器的输出物品：{}".format(text, output_text))
                if not self._parse_machines(machine_text):
                    logging.error("无法从字符串({})中解析机器列表：{}".format(text, machine_text))
                    return False
                return True
            return False
        except Exception as e:
            logging.warning("无法从字符串({})中解析机器列表数据：{}".format(text, e))
            return False

    @staticmethod
    def _mermaid_ex_work(begin_name: str, current_name: str, machine: 'MachineList', result: list[str]):
        inputs = machine.inputs.copy()
        inputs.pop_power()
        result.append(mermaid_link(begin_name, current_name, inputs))

    def _mermaid_parallel(self, namer: NameIterator) -> str:
        if not self.machines:
            return ""
        result = []
        if ALLOW_MERMAID_COMMENT:
            result.append("%% 开始并联 {}".format(self.name))
        if ALLOW_PARALLEL_GROUP_SUBGRAPH:
            result.append("subgraph {}".format(self.name))
        #last_output = namer.current
        parallel_input = namer.step()
        result.append(mermaid_element(parallel_input, "{}并联输入".format(self.name)))
        #parallel_inputs = self.inputs.copy()
        #parallel_inputs.pop_power()
        #result.append(mermaid_link(last_output, parallel_input, parallel_inputs))
        
        name_and_output_list = list()
        for machine in self.machines:
            current_name = namer.peek()
            if isinstance(machine, MachineList):
                result.append(machine.mermaid(namer))
                MachineList._mermaid_ex_work(parallel_input, current_name, machine, result)
            else:
                current_name = namer.step()
                result.append(mermaid_element(current_name, machine.name))
                result.append(mermaid_link(parallel_input, current_name, machine.inputs))
            end_name = namer.current
            name_and_output_list.append((end_name, machine.outputs))

        output_name = namer.step()
        result.append(mermaid_element(output_name, "{}并联输出".format(self.name)))
        for end_name, outputs in name_and_output_list:
            result.append(mermaid_link(end_name, output_name, outputs))
        if ALLOW_PARALLEL_GROUP_SUBGRAPH:
            result.append("end")
        if ALLOW_MERMAID_COMMENT:
            result.append("%% 结束并联 {}".format(self.name))
        return '\n\t'.join(result)
    
    def _mermaid_serial(self, namer: NameIterator) -> str:
        if not self.machines:
            return ""
        
        result = []
        if ALLOW_MERMAID_COMMENT:
            result.append("%% 开始串联 {}".format(self.name))
        if ALLOW_SERIAL_GROUP_SUBGRAPH:
            result.append("subgraph {}".format(self.name))

        serial_input = namer.step()
        result.append(mermaid_element(serial_input, "{}串联输入".format(self.name)))
        last_input = serial_input

        for machine in self.machines:
            current_name = namer.peek()
            if isinstance(machine, MachineList):
                result.append(machine.mermaid(namer))
                MachineList._mermaid_ex_work(last_input, current_name, machine, result)
            else:
                namer.step()
                result.append(mermaid_element(current_name, machine.name))
                result.append(mermaid_link(last_input, current_name, machine.inputs))
            last_input = namer.current
            
        serial_output = namer.step()
        result.append(mermaid_element(serial_output, "{}串联输出".format(self.name)))
        result.append(mermaid_link(last_input, serial_output, self.outputs))
        if ALLOW_SERIAL_GROUP_SUBGRAPH:
            result.append("end")
        if ALLOW_MERMAID_COMMENT:
            result.append("%% 结束串联 {}".format(self.name))
        return '\n\t'.join(result)

    def mermaid(self, namer: NameIterator) -> str:
        if self.composition == Composition.parallel.value:
            return self._mermaid_parallel(namer)
        elif self.composition == Composition.serial.value:
            return self._mermaid_serial(namer)
        raise ValueError("未知的组合类型：{}".format(self.composition))
            
    def __len__(self):
        return len(self.machines)
    
    def __getitem__(self, index: int) -> Machine | None:
        if 0 <= index < len(self.machines):
            return self.machines[index]
        return None
    
    def __iter__(self):
        return iter(self.machines)
    
    def __str__(self):
        return "{}<{}>({})".format(
            Machine.__str__(self),
            self.composition, 
            ';'.join(str(machine) for machine in self.machines))
        
class Parallel(MachineList):
    def __init__(self, name: str = "并联机器", machines: list[Machine] = list()):
        if not machines:
            machines = list()
        super().__init__(
            name,
            ItemList(),
            ItemList(),
            ItemList(),
            machines, 
            Composition.parallel.value)
        self.reinit()
        
    def borrow(self, machines: MachineList) -> bool:
        if machines.composition != Composition.parallel.value:
            return False
        
        self.name = machines.name
        self.cost = machines.cost
        self.inputs = machines.inputs
        self.outputs = machines.outputs
        self.machines = machines.machines
        return True

class Serial(MachineList):
    def __init__(self, name: str = "串联机器", machines: list[Machine] = list()):
        if not machines:
            machines = list()
        super().__init__(
            name, 
            ItemList(),
            ItemList(),
            ItemList(),
            machines, 
            Composition.serial.value)
        
        try:
            self.reinit()
        except Exception as e:
            logging.error("串联机器组({})构建失败；{}".format(self.name, e))

    def borrow(self, machines: MachineList) -> bool:
        if machines.composition != Composition.serial.value:
            return False

        self.name = machines.name
        self.cost = machines.cost
        self.inputs = machines.inputs
        self.outputs = machines.outputs
        self.machines = machines.machines
        return True

def multiple(machine: Machine|Parallel|Serial, factor: float) -> Machine:
    if factor <= 0:
        return Machine('0x{}'.format(machine.name))
    
    if factor.is_integer() and int(factor) == 1:
        return machine.copy()
    
    machines = []
    value = factor
    while value > 1:
        machines.append(machine.copy())
        value -= 1
    last = machine.copy()
    last.inputs.multiply(value)
    last.outputs.multiply(value)
    machines.append(last)
    return Parallel('{}x{}'.format(FloatStr(factor), machine.name), machines)

def mermaid(machine: MachineList) -> str:
    result = ["graph TB"]
    namer = NameIterator()
    begin_name = namer.current
    result.append(mermaid_element(begin_name, "开始"))
    postwork = None
    if isinstance(machine, MachineList):
        parallel_input = namer.peek()
        parallel_inputs = machine.inputs.copy()
        parallel_inputs.pop_power()
        postwork = mermaid_link(begin_name, parallel_input, parallel_inputs)
    
    result.append(machine.mermaid(namer))
    if postwork is not None:
        result.append(postwork)
    return '\n\t'.join(result)

#endregion

#region IO

def serialize(obj: Item|ItemList|Machine|MachineList|list) -> str:
    if isinstance(obj, list):
        result = list()
        for item in obj:
            result.append(serialize(item))
        return "".join(result)

    prefix = "#"
    if isinstance(obj, Item):
        prefix = "Item"
    if isinstance(obj, ItemList):
        prefix = "ItemList"
    if isinstance(obj, Machine):
        prefix = "Machine"
    if isinstance(obj, MachineList):
        prefix = "MachineList"
        
    return "{} {}\n".format(prefix, obj)

class LookUpTable:
    def __init__(self):
        self.table : dict[str, str] = {}
        
    def lookup(self, input: str) -> str:
        pattern = r'\$\(([^)]+)\)'
        target_keys = re.findall(pattern, input)
        if not target_keys:
            return input
        missing = [key for key in target_keys if key not in self.table]
        if missing:
            message = "在查找表中未找到以下键：{}".format(missing)
            logging.error(message)
            raise KeyError(message)
        def replace_match(match):
            key = match.group(1)
            return self.table[key]
        return re.sub(pattern, replace_match, input)
    
    def insert(self, obj: Machine):
        self.table[obj.name] = str(obj)

class MachineTable:
    def __init__(self):
        self.table: dict[str, Machine] = {}
        
    def load(self, objs: list) -> 'MachineTable':
        for obj in objs:
            if isinstance(obj, Machine):
                self.insert(obj)
                logging.info("已加载机器：{}".format(obj.name))
        return self

    def lookup(self, name: str) -> Machine | None:
        return self.table.get(name)

    def insert(self, obj: Machine):
        self.table[obj.name] = obj

    def tolist(self) -> list[Machine]:
        return list(self.table.values())
    
    def lookup_adv(self, content) -> str:
        for key in self.table.keys():
            if content in key:
                return key
        return ""
    
    def __getitem__(self, name: str) -> Machine:
        machine = self.table.get(name)
        if not machine:
            key = self.lookup_adv(name)
            if not key:
                message = "未找到机器：{}".format(name)
                logging.error(message)
                raise KeyError(message)
            else:
                logging.info("未找到目标机器({})，但是搜索到关联机器({})".format(name, key))
            return self.table[key].copy()
        return machine.copy()
    
    def __call__(self, name: str) -> Machine:
        return self[name]

def deserilize(lines: str, table: LookUpTable = LookUpTable()) -> list[Item|ItemList|Machine|MachineList]:
    result = list()
    failed_count = 0
    succeeded_count = 0
    for line in lines.splitlines():
        try:
            if not line or line.startswith("#"):
                continue
            
            logging.info("正在解析行：{}".format(line))
            
            if line.startswith("Item "):
                item = Item().copy()
                if item.parse(line[5:]):
                    succeeded_count += 1
                    result.append(item)
                    logging.info("成功解析Item：{}".format(item))
                else:
                    failed_count += 1
                continue
            
            if line.startswith("ItemList "):
                items = ItemList().copy()
                if items.parse(line[9:]):
                    succeeded_count += 1
                    result.append(items)
                    logging.info("成功解析ItemList：{}".format(items))
                else:
                    failed_count += 1
                continue
            
            if line.startswith("Machine "):
                machine = Machine().copy()
                if machine.parse(table.lookup(line[8:])):
                    succeeded_count += 1
                    result.append(machine)
                    table.insert(machine)
                    logging.info("成功解析Machine：{}".format(machine))
                else:
                    failed_count += 1
                continue
                
            if line.startswith("MachineList "):
                machine_list = MachineList().copy()
                if machine_list.parse(table.lookup(line[12:])):
                    succeeded_count += 1
                    result.append(machine_list)
                    table.insert(machine_list)
                    logging.info("成功解析MachineList：{}".format(machine_list))
                else:
                    failed_count += 1
                continue
                
            logging.warning("无法从字符串({})中解析对象数据".format(line))

        except Exception as e:
            logging.warning("无法从字符串({})中解析对象数据：{}".format(line, e))
            continue
    return result

def readlines(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.warning("无法打开文件并读取({})：{}".format(path, e))
        return ""
    
def format_architect(architect: str) -> str:
    result = ""
    level: int = 0
    tab = ' ' * 4
    last_char = ''
    for char in architect:
        if char == '(':
            level += 1
            result += char + '\n' + tab * level
            continue
        if char == ')':
            level -= 1
            if level < 0:
                raise ValueError("括号不匹配")
            result += '\n' + tab * level + char
            continue
        if char == '|' and last_char == '':
            last_char = char
            continue
        if char == '|' and last_char == '|':
            last_char = ''
            result += '||\n' + tab * level
            continue
        if char == '-' and last_char == '':
            last_char = '-'
            continue
        if char == '>' and last_char == '-':
            result += '->\n' + tab * level
            last_char = ''
            continue
        last_char = ''
        result += char
    return result

def save_text(text: str, path: str, append: bool = False):
    try:
        open_method = 'a' if append else 'w'
            
        with open(path, open_method, encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        logging.warning("无法打开文件并写入({})：{}".format(path, e))

#endregion

if __name__ == "__main__":
    # [level] message ...
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    objs = deserilize(readlines("data.txt"))
    machine_table = MachineTable().load(objs)
    
    save_text(serialize(machine_table.tolist()), "out/out.txt")
    