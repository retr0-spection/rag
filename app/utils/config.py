mermaid_config = '''# Mermaid Diagram Formats

Mermaid diagrams are defined using a simple, text-based syntax. Here are examples of common diagram types:

## 1. Flowchart (NOT A BAR GRAPH!!!)

```mermaid
graph TD
    A[Start] --> B{{Is it?}}
    B -->|Yes| C[OK]
    C --> D[Rethink]
    D --> B
    B ---->|No| E[End]
```
and
```mermaid
graph TD
          A[Christmas] -->|Get money| B(Go shopping)
          B --> C{Let me think}
          B --> G[/Another/]
          C ==>|One| D[Laptop]
          C -->|Two| E[iPhone]
          C -->|Three| F[fa:fa-car Car]
          subgraph section
            C
            D
            E
            F
            G
          end
```

## 2. Sequence Diagram

```mermaid
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop Healthcheck
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail!
    John-->>Alice: Great!
    John->>Bob: How about you?
    Bob-->>John: Jolly good!
```

## 3. Gantt Chart

```mermaid
gantt
    title A Gantt Diagram
    dateFormat  YYYY-MM-DD
    section Section
    A task           :a1, 2014-01-01, 30d
    Another task     :after a1  , 20d
    section Another
    Task in sec      :2014-01-12  , 12d
    another task      : 24d
```

## 4. Class Diagram

```mermaid
classDiagram
    Animal <|-- Duck
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class Duck{{
      +String beakColor
      +swim()
      +quack()
    }}
    class Fish{{
      -int sizeInFeet
      -canEat()
    }}
    class Zebra{{
      +bool is_wild
      +run()
    }}
```

## 5. State Diagram

```mermaid
stateDiagram-v2
    [*] --> Still
    Still --> [*]
    Still --> Moving
    Moving --> Still
    Moving --> Crash
    Crash --> [*]
```

## 6. ERD Diagram

```mermaid
---
title: Order example
---
erDiagram
    CUSTOMER ||--o{{ ORDER : places
    ORDER ||--|{{ LINE-ITEM : contains
    CUSTOMER }}|..|{{ DELIVERY-ADDRESS : uses
```

## 7. User Journey Diagram

```mermaid
journey
    title My working day
    section Go to work
      Make tea: 5: Me
      Go upstairs: 3: Me
      Do work: 1: Me, Cat
    section Go home
      Go downstairs: 5: Me
      Sit down: 5: Me
```

## 8. Sankey
Syntax
The idea behind syntax is that a user types sankey-beta keyword first, then pastes raw CSV below and get the result.

It implements CSV standard as described here with subtle differences:

CSV must contain 3 columns only
It is allowed to have empty lines without comma separators for visual purposes

Commas
If you need to have a comma, wrap it in double quotes:
```mermaid
sankey-beta
%% source,target,value
Pumped heat,"Heating and cooling, homes",193.026
Pumped heat,"Heating and cooling, commercial",70.672
```
See how there's still 3 values in each line

For numbers
Example:
    Wrong - 250,000
    Correct - 250000

num_headings === num_entries

Each ',' separates an entry hence be mindful of how you use them
Do not add more entries than headers
DO NOT DEVIATE from format prefixed by %%
3 values in each line seperated by ,


```mermaid
sankey-beta

%% source,target,value
Electricity grid,Over generation / exports, 104.453
Electricity grid,Heating and cooling - homes, 113.726
Electricity grid,H2 conversion, 27.14
```

## 9. XY Chart (alias Bar Graph, Line Graph, Histogram)
# note that his can be used to generate Bar Graphs, Line Graphs, Histograms
```mermaid
---
config:
    xyChart:
        width: 900
        height: 600
    themeVariables:
        xyChart:
            titleColor: "#ff0000"
---
xychart-beta
    title "Sales Revenue"
    x-axis [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
    y-axis "Revenue (in $)" 4000 --> 11000
    bar [5000, 6000, 7500, 8200, 9500, 10500, 11000, 10200, 9200, 8500, 7000, 6000]
    line [5000, 6000, 7500, 8200, 9500, 10500, 11000, 10200, 9200, 8500, 7000, 6000]
```

## 10. Architecture

```mermaid
architecture-beta
    service left_disk(disk)[Disk]
    service top_disk(disk)[Disk]
    service bottom_disk(disk)[Disk]
    service top_gateway(internet)[Gateway]
    service bottom_gateway(internet)[Gateway]
    junction junctionCenter
    junction junctionRight

    left_disk:R -- L:junctionCenter
    top_disk:B -- T:junctionCenter
    bottom_disk:T -- B:junctionCenter
    junctionCenter:R -- L:junctionRight
    top_gateway:B -- T:junctionRight
    bottom_gateway:T -- B:junctionRight
```

and
```mermaid
architecture-beta
    group api(cloud)[API]

    service db(database)[Database] in api
    service disk1(disk)[Storage] in api
    service disk2(disk)[Storage] in api
    service server(server)[Server] in api

    db:L -- R:server
    disk1:T -- B:server
    disk2:T -- B:db
```

## 11. Mindmaps

```mermaid
mindmap
  root((mindmap))
    Origins
      Long history
      ::icon(fa fa-book)
      Popularisation
        British popular psychology author Tony Buzan
    Research
      On effectiveness<br/>and features
      On Automatic creation
        Uses
            Creative techniques
            Strategic planning
            Argument mapping
    Tools
      Pen and paper
      Mermaid
```

You can include mathematics by using KaTex Typesetter using Flowcharts and Sequences ex.
```mermaid
graph LR
     A["$$x^2$$"] -->|"$$\sqrt{{x+3}}$$"| B("$$\frac{{1}}{{2}}$$")
     A -->|"$$\overbrace{{a+b+c}}^{{\text{{note}}}}$$"| C("$$\pi r^2$$")
     B --> D("$$x = \begin{{cases}} a &\text{{if }} b \\ c &\text{{if }} d \end{{cases}}$$")
     C --> E("$$x(t)=c_1\begin{{bmatrix}}-\cos{{t}}+\sin{{t}}\\ 2\cos{{t}} \end{{bmatrix}}e^{{2t}}$$")
```

and

```mermaid
sequenceDiagram
    autonumber
    participant 1 as $$\alpha$$
    participant 2 as $$\beta$$
    1->>2: Solve: $$\sqrt{{2+2}}$$
    2-->>1: Answer: $$2$$
    Note right of 2: $$\sqrt{{2+2}}=\sqrt{{4}}=2$$
```


When using these in your chat application, the content should be wrapped in a code block with the language specified as 'mermaid'. For example:

    ```mermaid
    graph TD
        A[Start] --> B{{Is it?}}
        B -->|Yes| C[OK]
        B -->|No| D[End]
    ```

BECAREFUL WITH FORMAT.
Avoid errors like this: Expecting 'SPACE', 'AMP', 'COLON', 'DOWN', 'DEFAULT', 'NUM', 'COMMA', 'NODE_STRING', 'BRKT', 'MINUS', 'MULT', 'UNICODE_TEXT', got 'NEWLINE'
Avoid errors like this:-->|14.29%| Q4[Q4
----------------------^
Expecting 'SQE', 'DOUBLECIRCLEEND', 'PE', '-)', 'STADIUMEND', 'SUBROUTINEEND', 'PIPE', 'CYLINDEREND', 'DIAMOND_STOP', 'TAGEND', 'TRAPEND', 'INVTRAPEND', 'UNICODE_TEXT', 'TEXT', 'TAGSTART', got '1'
This format ensures that your MermaidRenderer component can properly process and render the diagram.'''
