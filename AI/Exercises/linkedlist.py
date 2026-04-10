class Node:
    def __init__(self, value):
        self.value = value
        self._next = None
        self._prev = None


class DoublyLinkedList:
    def __init__(self):
        self._head = None
        self._tail = None
        self._N = 0

    def add_first(self, value):
        new_node = Node(value)
        if not self._head:
            self._head = new_node
            self._tail = new_node
        else:
            new_node._next = self._head
            self._head._prev = new_node
            self._head = new_node
        self._N += 1

    def add_last(self, value):
        new_node = Node(value)
        if not self._tail:
            self._head = new_node
            self._tail = new_node
        else:
            new_node._prev = self._tail
            self._tail._next = new_node
            self._tail = new_node
        self._N += 1

    def remove_first(self):
        ret = None
        if self._head:
            ret = self._head.value
            if self._head is self._tail:
                self._head = None
                self._tail = None
            else:
                self._head = self._head._next
                self._head._prev = None
            self._N -= 1
        return ret

    def remove_last(self):
        ret = None
        if self._tail:
            ret = self._tail.value
            if self._head is self._tail:
                self._head = None
                self._tail = None
            else:
                self._tail = self._tail._prev
                self._tail._next = None
            self._N -= 1
        return ret

    def __str__(self):
        s = "DoublyLinkedList: "
        node = self._head
        while node:
            s += "{} ==> ".format(node.value)
            node = node._next
        return s

    def __len__(self):
        return self._N


if __name__ == "__main__":
    L = DoublyLinkedList()
    L.add_first(10)
    L.add_first(4)
    L.add_first("chris")
    L.add_last("layla")
    L.add_last("theo")
    print(L)
    print(len(L))
    print(L.remove_first())
    print(L.remove_last())
    print(L)
    print(len(L))