import dataclasses

from tree_sitter import Language, Node, Parser
import tree_sitter_cpp

class FeatureExtractor:

    @dataclasses.dataclass
    class Features:
        loc: int = 0
        branch_count: int = 0
        operand_count: int = 0
        operator_count: int = 0
        unique_operand_count: int = 0
        unique_operator_count: int = 0
        condition_operand_count: int = 0
        array_operand_count: int = 0
        pointer_expression_count: int = 0
        arithmetic_expression_count: int = 0
        constant_count: int = 0

    language = Language(tree_sitter_cpp.language())
    expression_node = 'expression_statement'
    branch_nodes = {
        'if_statement', 'switch_statement', 'do_statement',
        'while_statement', 'for_statement', 'for_range_loop',
        'try_statement', 'seh_try_statement', 'throw_statement',
        'goto_statement', 'co_yield_statement', 'break_statement',
        'continue_statement', '&&', '||', 'conditional_expression'
    }
    identifier_node = 'identifier'
    condition_field = 'condition'
    operand_nodes = {
        'identifier', 'type_identifier', 'namespace_identifier',
        'field_identifier', 'number_literal', 'string_literal',
        'char_literal', 'true', 'false', 'null', 'this'
    }
    arithmetic_nodes = {'binary_expression', 'unary_expression'}
    constant_node = 'number_literal'
    operator_field = 'operator'

    def __init__(self):
        self.parser = Parser(self.language)

    def extract(self, function: list[str]) -> Features:
        self.features = self.Features(loc=len(function))
        self.operands = set()
        self.operators = set()
        for line in function:
            tree = self.parser.parse(line.encode())
            self._process_node(tree.root_node, cond=False, array=False)
        self.features.operand_count -= \
            self.features.condition_operand_count + \
            self.features.array_operand_count
        return self.features

    def _process_node(self, node: Node, cond: bool, array: bool):
        if node.type in self.branch_nodes:
            self.features.branch_count += 1
        if node.type in self.operand_nodes:
            self.features.operand_count += 1
            self.features.condition_operand_count += cond
            self.features.array_operand_count += array
            assert node.parent is not None
            if node.text not in self.operands:
                self.features.unique_operand_count += 1
                self.operands.add(node.text)
        if ('pointer' in node.type or 'array' in node.type or
            'sizeof' in node.type or 'subscript' in node.type):
            self.features.pointer_expression_count += 1
            array = True
        if node.type in self.arithmetic_nodes:
            self.features.arithmetic_expression_count += 1
        if node.type == self.constant_node:
            self.features.constant_count += 1

        operator = node.child_by_field_name(self.operator_field)
        if operator is not None:
            self.features.operator_count += 1
            if operator.text not in self.operators:
                self.features.unique_operator_count += 1
                self.operators.add(operator.text)

        for i in range(node.child_count):
            field = node.field_name_for_child(i)
            child = node.children[i]
            self._process_node(
                child, cond or field == self.condition_field, array)
