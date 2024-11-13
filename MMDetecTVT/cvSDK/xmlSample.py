import BasicUseFunc as basFunc
import os
import os.path
import xml.dom.minidom





def get_xml_dom(xml_file=''):
    """
    """
    if xml_file:
        dom = xml.dom.minidom.parse(xml_file)
    else:
        dom = xml.dom.minidom.Document()
    return dom


def create_node(dom,new_node,parent_node=''):
    """
    创建节点(包括根节点)
    """
    if not parent_node:
        parent_node = dom.createElement('root')
        dom.appendChild(parent_node)


    node = dom.createElement(new_node)
    parent_node.appendChild(node)
    return node


def create_node_text(dom,node,txt):
    """
    节点赋值
    """
    txt = dom.createTextNode(txt)
    node.appendChild(txt)
    return node


def get_node_value(node,tag):
    """
    """
    for x in node:
        nm = x.nodeName        
        if nm == tag:
            return x.firstChild.data
    return None
def get_node_data(node):
    """
    获得节点文本
    """
    return node.firstChild.data


def set_node_data(node,txt):
    """
    节点赋值 只有节点上有值 才能 x.firstChild.data
    """
    for x in node:
        x.firstChild.data = txt
    return node


def get_node_name(node):
    """
    获得节点名
    """
    return node.nodeName


def set_node_attribute(node,key,val):
    """
    设置节点属性
    """
    node.setAttribute(key,val)
    return node


def remove_node_child(parent_node,c_node):
    """
    删除子节点
    """
    return parent_node.removeChild(c_node)


def get_elementsByTagName(dom,tag):
    """
    获取标签
    """
    root = dom
    return root.getElementsByTagName(tag)

    
def get_node_dataEx(dom,tag):
    node=get_elementsByTagName(dom,tag)
    if(node):
        return get_node_data(node[0])
    return None


def write_xml(dom='',new_xml_file='new_xml.xml'):
    """
    生成xml
    """
    if not dom:
        dom = get_xml_dom()


    #.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
    # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
    with open(new_xml_file,'w') as fx:
        dom.writexml(fx,indent='',addindent='\t',newl='\n',encoding='UTF-8')

