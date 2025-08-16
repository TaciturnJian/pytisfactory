from dataclasses import dataclass
from enum import Enum
import re
import logging

#region Game Elements

@dataclass
class Item:
    """
    satisfactory 中的一个物品，包含名称、数量和仅用于格式化的名称后缀
    """
    name: str = "物品"
    count: float = 0.0
    postfix: str = "单位"
    
    def equal(self, other: 'Item|str') -> bool:
        if isinstance(other, str):
            other = Item(other)
        
        return self.name == other.name and self.count == other.count
    
    def has_name(self, name: str) -> bool:
        return self.name == name
    
    def simplify(self) -> 'Item':
        self.name = self.name.strip()
        self.postfix = self.postfix.strip()
        return self
    
    def parse(self, text: str) -> bool:
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
        return "{}({}{})".format(self.name, self.count, self.postfix)

class Power(Item):
    def __init__(self, count: float = 0):
        super().__init__(
            "电力", 
            count, 
            "MW")

class ItemList:
    def __init__(self, items: list[Item] = list()):
        if not items:
            items = list()
        self.items = items        
        
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
            if my_item.has_name(item):
                return True
        return False
    
    def index_of(self, item: Item|str) -> int:
        if isinstance(item, Item):
            item = item.name
        for i, my_item in enumerate(self.items):
            if my_item.has_name(item):
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
        
    def multiply(self, factor: float) -> 'Machine':
        self.name = "{}x{}".format(factor, self.name)
        self.cost.multiply(factor)
        self.inputs.multiply(factor)
        self.outputs.multiply(factor)
        return self

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

class Composition(Enum):
    unknown = "未知"
    serial = "串联"
    parallel = "并联"

class NameIterator:
    def __init__(self, begin: str = 'a'):
        self._current = begin
    
    def current(self):
        return self._current
    
    def next(self):
        return self._increment(self._current)
    
    def step(self):
        self._current = self._increment(self._current)
        return self._current
    
    def _increment(self, s):
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
    name = name.strip()
    label = label.strip()
    if label != "":
        label = "[\"{}\"]".format(label)
    return "{}{}".format(name, label)

def mermaid_link(from_id: str, to_id: str, comment: str|ItemList = "") -> str:
    if isinstance(comment, ItemList):
        comment = str(comment)
    comment = comment.strip()
    if comment != "":
        comment = "|\"{}\"|".format(comment)
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
            [machine.copy() for machine in self.machines]
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

    def _mermaid_parallel(self, namer: NameIterator) -> str:
        result = ["subgraph {}".format(self.name)]
        input_name = namer.current()
        parallel_input = namer.step()
        result.append(mermaid_element(parallel_input, "{}输入".format(self.name)))
        parallel_inputs = self.inputs.copy()
        parallel_inputs.pop_power()
        result.append(mermaid_link(input_name, parallel_input, parallel_inputs))
        
        name_and_output_list = list()
        for machine in self.machines:
            current_name = namer.next()
            if isinstance(machine, MachineList):
                result.append(machine.mermaid(namer))
            else:
                namer.step()
                result.append(mermaid_element(current_name, machine.name))
                result.append(mermaid_link(parallel_input, current_name, machine.inputs))
            end_name = namer.current()
            name_and_output_list.append((end_name, machine.outputs))

        output_name = namer.step()
        result.append(mermaid_element(output_name, "{}输出".format(self.name)))
        for end_name, outputs in name_and_output_list:
            result.append(mermaid_link(end_name, output_name, outputs))
        result.append("end")
        return '\n\t'.join(result)
    
    def _mermaid_serial(self, namer: NameIterator) -> str:
        #result = ["subgraph {}".format(self.name)]
        result = []
        input_name = namer.current()
        
        serial_input = namer.step()
        result.append(mermaid_element(serial_input, "{}输入".format(self.name)))
        serial_inputs = self.inputs.copy()
        serial_inputs.pop_power()
        result.append(mermaid_link(input_name, serial_input, serial_inputs))

        last_input = serial_input

        for machine in self.machines:
            current_name = namer.next()
            if isinstance(machine, MachineList):
                result.append(machine.mermaid(namer))
            else:
                namer.step()
                result.append(mermaid_element(current_name, machine.name))
                result.append(mermaid_link(last_input, current_name, machine.inputs))
            last_input = namer.current()
            
        serial_output = namer.step()
        result.append(mermaid_element(serial_output, "{}输出".format(self.name)))
        result.append(mermaid_link(last_input, serial_output, self.outputs))
        #result.append("end")
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
    def __init__(self, name: str = "并行机器", machines: list[Machine] = list()):
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
    def __init__(self, name: str = "串行机器", machines: list[Machine] = list()):
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

def mermaid(machine: MachineList) -> str:
    result = ["graph LR"]
    namer = NameIterator()
    result.append(mermaid_element(namer.current(), "开始"))
    result.append(machine.mermaid(namer))
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
    