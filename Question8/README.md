For this question, python 3.11 was used. Prevous versions of python may work, but only 3.11 has been tested on these questions.

The dataset used by this question is a subset of the ImageNet ILSVRC 2012 dataset. See the README in the `datasets/imagenet` directory for information on how to download the dataset. These files must be downloaded and placed in the `datasets/imagenet` directory before running the code in this question and the dependencies in `requirements.txt` must be installed. For best results, it is recommended to manually install pytorch first.

If you don't want to worry about extracting the imagenet archive files, the alexnet.py file will extract the archives as part of the data loading process. After this step has completed, yolo will then be able to run the imagenet dataset.

## Running Each Question

### Part a:

Run `python alexnet.py` from the `Question8` directory after downloading the needed imagenet files and after installing the dependencies listed in the repository's `requirements.txt`.

### Part b and c

The yolo models can be run with the appropratie script:
- For yolov8: `./yolov8.sh` or `.\yolov8.bat` on windows should be run from the `Question8` directory
- For yolov5: `./yolov5.sh` or `.\yolov5.bat` on windows should be run from the `Question8` directory
    - Make sure you cloned the repository with `git clone --recurse-submodules` to download yolo5

Note: If YOLO complains about not being able to find the dataset, try running the alexnet script first. This script also handles unpacking the tarballs for the imagenet dataset.

Pi Run Results-
- Unfortunately, for both yolov5 and yolov8, yolo just segfaults on our pi 3's. - RIP

Laptop Specs:
- CPU: Intel Core i7-8650U @ 2.11 GHz base clock
- GPU: NVIDIA GeForce GTX 1050

YOLOv8 Run Results for Laptop:

```
yolo val model=yolov8s-cls.pt data=imagenet batch=1 imgsz=224 plots=true save_json=true 
Ultralytics YOLOv8.0.208  Python-3.11.5 torch-2.1.0 CUDA:0 (GeForce GTX 1050, 2048MiB)
YOLOv8s-cls summary (fused): 73 layers, 6356200 parameters, 0 gradients, 13.5 GFLOPs
train: C:\Users\kytec\source\repos\USU\CS5510GroupOopsAllGrads\Question8\datasets\imagenet\train... found 1 images in 1 classes

val: C:\Users\kytec\source\repos\USU\CS5510GroupOopsAllGrads\Question8\datasets\imagenet\val... found 50000 images in 1000 classes: ERROR  requires 1 classes, not 1000
test: None...
val: Scanning C:\Users\kytec\source\repos\USU\CS5510GroupOopsAllGrads\Question8\datasets\imagenet\val... 50000 images, 0 corru
               classes   top1_acc   top5_acc: 100%|██████████| 50000/50000 [20:47<00:00, 40.09it/s]  
                   all      0.677      0.881
Speed: 0.5ms preprocess, 10.3ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to runs\classify\val6
 Learn more at https://docs.ultralytics.com/modes/val
```

YOLOv5 Run Results for Laptop

```
YOLOv5  2023-11-10 Python-3.11.5 torch-2.1.0 CUDA:0 (GeForce GTX 1050, 2048MiB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt to yolov5s-cls.pt...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 10.5M/10.5M [00:00<00:00, 18.4MB/s]

Fusing layers... 
Model summary: 117 layers, 5447688 parameters, 0 gradients, 11.4 GFLOPs
validating: 100%|██████████| 50000/50000 [26:16<00:00, 31.71it/s]  
                   Class      Images    top1_acc    top5_acc
                     all       50000       0.715       0.902
                   tench          50        0.94        0.98
                goldfish          50        0.88        0.92
       great white shark          50        0.78        0.96
             tiger shark          50        0.68        0.96
        hammerhead shark          50        0.82        0.92
            electric ray          50        0.76         0.9
                stingray          50         0.7         0.9
                    cock          50        0.78        0.92
                     hen          50        0.84        0.96
                 ostrich          50        0.98           1
               brambling          50         0.9        0.96
               goldfinch          50        0.92        0.98
             house finch          50        0.88        0.96
                   junco          50        0.94        0.98
          indigo bunting          50        0.86        0.88
          American robin          50         0.9        0.96
                  bulbul          50        0.84        0.96
                     jay          50         0.9        0.96
                  magpie          50        0.84        0.96
               chickadee          50         0.9           1
         American dipper          50        0.82        0.92
                    kite          50        0.76        0.94
              bald eagle          50        0.92           1
                 vulture          50        0.96           1
          great grey owl          50        0.94        0.98
         fire salamander          50        0.96        0.98
             smooth newt          50        0.58        0.94
                    newt          50        0.74         0.9
      spotted salamander          50        0.86        0.94
                 axolotl          50        0.86        0.96
       American bullfrog          50        0.78        0.92
               tree frog          50        0.84        0.96
             tailed frog          50        0.48         0.8
   loggerhead sea turtle          50        0.68        0.94
  leatherback sea turtle          50         0.5         0.8
              mud turtle          50        0.64        0.84
                terrapin          50        0.52        0.98
              box turtle          50        0.84        0.98
            banded gecko          50         0.7        0.88
            green iguana          50        0.76        0.94
          Carolina anole          50        0.58        0.96
desert grassland whiptail lizard          50        0.82        0.94
                   agama          50        0.74        0.92
   frilled-necked lizard          50        0.84        0.86
        alligator lizard          50        0.58        0.78
            Gila monster          50        0.72         0.8
   European green lizard          50        0.42         0.9
               chameleon          50        0.76        0.84
           Komodo dragon          50        0.86        0.96
          Nile crocodile          50         0.7        0.84
      American alligator          50        0.76        0.96
             triceratops          50         0.9        0.94
              worm snake          50        0.76        0.88
       ring-necked snake          50         0.8        0.92
 eastern hog-nosed snake          50        0.58        0.88
      smooth green snake          50         0.6        0.94
               kingsnake          50        0.82         0.9
            garter snake          50        0.88        0.94
             water snake          50         0.7        0.94
              vine snake          50        0.66        0.76
             night snake          50        0.32        0.82
         boa constrictor          50         0.8        0.96
     African rock python          50        0.48        0.76
            Indian cobra          50        0.82        0.94
             green mamba          50        0.54        0.86
               sea snake          50        0.62         0.9
    Saharan horned viper          50        0.56        0.86
eastern diamondback rattlesnake          50         0.6        0.86
              sidewinder          50        0.26        0.86
               trilobite          50        0.98        0.98
              harvestman          50        0.86        0.94
                scorpion          50        0.86        0.94
    yellow garden spider          50        0.92        0.96
             barn spider          50        0.38        0.98
  European garden spider          50        0.62        0.98
    southern black widow          50        0.88        0.94
               tarantula          50        0.94           1
             wolf spider          50        0.82        0.92
                    tick          50        0.74        0.84
               centipede          50        0.68        0.82
            black grouse          50        0.88        0.98
               ptarmigan          50        0.78        0.94
           ruffed grouse          50        0.88           1
          prairie grouse          50        0.92           1
                 peacock          50        0.88         0.9
                   quail          50         0.9        0.94
               partridge          50        0.74        0.96
             grey parrot          50         0.9        0.96
                   macaw          50        0.88        0.98
sulphur-crested cockatoo          50        0.86        0.92
                lorikeet          50        0.96           1
                  coucal          50        0.82        0.88
               bee eater          50        0.96        0.98
                hornbill          50         0.9        0.96
             hummingbird          50        0.88        0.96
                 jacamar          50        0.92        0.94
                  toucan          50        0.84        0.94
                    duck          50        0.76        0.94
  red-breasted merganser          50        0.86        0.96
                   goose          50        0.74        0.96
              black swan          50        0.94        0.98
                  tusker          50        0.54        0.92
                 echidna          50        0.98           1
                platypus          50        0.72        0.84
                 wallaby          50        0.78        0.88
                   koala          50        0.84        0.92
                  wombat          50        0.78        0.84
               jellyfish          50        0.88        0.96
             sea anemone          50        0.72         0.9
             brain coral          50        0.88        0.96
                flatworm          50         0.8        0.98
                nematode          50        0.86         0.9
                   conch          50        0.74        0.88
                   snail          50        0.78        0.88
                    slug          50        0.74        0.82
                sea slug          50        0.88        0.98
                  chiton          50        0.88        0.98
      chambered nautilus          50        0.88        0.92
          Dungeness crab          50        0.78        0.94
               rock crab          50        0.68        0.86
            fiddler crab          50        0.64        0.86
           red king crab          50        0.76        0.96
        American lobster          50        0.78        0.96
           spiny lobster          50        0.74        0.88
                crayfish          50        0.56        0.86
             hermit crab          50        0.76        0.96
                  isopod          50        0.66        0.78
             white stork          50        0.88        0.96
             black stork          50        0.84        0.98
               spoonbill          50        0.96           1
                flamingo          50        0.94           1
       little blue heron          50        0.92        0.98
             great egret          50         0.9        0.96
                 bittern          50        0.86        0.94
            crane (bird)          50        0.62         0.9
                 limpkin          50        0.98           1
        common gallinule          50        0.92        0.96
           American coot          50         0.9        0.98
                 bustard          50        0.92        0.96
         ruddy turnstone          50        0.94           1
                  dunlin          50        0.86        0.94
         common redshank          50         0.9        0.96
               dowitcher          50        0.84        0.96
           oystercatcher          50        0.86        0.94
                 pelican          50        0.92        0.96
            king penguin          50        0.88        0.96
               albatross          50         0.9           1
              grey whale          50        0.84        0.92
            killer whale          50        0.92           1
                  dugong          50        0.84        0.96
                sea lion          50        0.82        0.92
               Chihuahua          50        0.66        0.84
           Japanese Chin          50        0.72        0.98
                 Maltese          50        0.76        0.94
               Pekingese          50        0.84        0.94
                Shih Tzu          50        0.74        0.96
    King Charles Spaniel          50        0.88        0.98
                Papillon          50        0.88        0.94
             toy terrier          50        0.48        0.94
     Rhodesian Ridgeback          50        0.76        0.98
            Afghan Hound          50        0.84           1
            Basset Hound          50         0.8        0.92
                  Beagle          50        0.82        0.96
              Bloodhound          50        0.48        0.72
      Bluetick Coonhound          50        0.86        0.94
 Black and Tan Coonhound          50        0.54         0.8
Treeing Walker Coonhound          50        0.66        0.98
        English foxhound          50        0.32        0.84
       Redbone Coonhound          50         0.6        0.94
                  borzoi          50        0.92           1
         Irish Wolfhound          50        0.48        0.88
       Italian Greyhound          50        0.76        0.98
                 Whippet          50        0.74        0.92
            Ibizan Hound          50         0.6        0.86
      Norwegian Elkhound          50        0.88        0.98
              Otterhound          50        0.62         0.9
                  Saluki          50        0.72        0.92
      Scottish Deerhound          50        0.86        0.98
              Weimaraner          50        0.88        0.94
Staffordshire Bull Terrier          50        0.66        0.98
American Staffordshire Terrier          50        0.64        0.92
      Bedlington Terrier          50         0.9        0.92
          Border Terrier          50        0.86        0.92
      Kerry Blue Terrier          50        0.78        0.98
           Irish Terrier          50         0.7        0.96
         Norfolk Terrier          50        0.68         0.9
         Norwich Terrier          50        0.72           1
       Yorkshire Terrier          50        0.66         0.9
        Wire Fox Terrier          50        0.64        0.98
        Lakeland Terrier          50        0.74        0.92
        Sealyham Terrier          50        0.76         0.9
        Airedale Terrier          50        0.82        0.92
           Cairn Terrier          50        0.76         0.9
      Australian Terrier          50        0.48        0.84
  Dandie Dinmont Terrier          50        0.82        0.92
          Boston Terrier          50        0.92           1
     Miniature Schnauzer          50        0.68         0.9
         Giant Schnauzer          50        0.72        0.98
      Standard Schnauzer          50        0.74           1
        Scottish Terrier          50        0.76        0.96
         Tibetan Terrier          50        0.48           1
Australian Silky Terrier          50        0.64        0.96
Soft-coated Wheaten Terrier          50        0.74        0.96
West Highland White Terrier          50        0.88        0.96
              Lhasa Apso          50        0.68        0.96
   Flat-Coated Retriever          50        0.72        0.94
  Curly-coated Retriever          50         0.8        0.94
        Golden Retriever          50        0.86        0.94
      Labrador Retriever          50        0.82        0.94
Chesapeake Bay Retriever          50        0.76        0.96
German Shorthaired Pointer          50         0.8        0.96
                  Vizsla          50        0.68        0.96
          English Setter          50         0.7           1
            Irish Setter          50         0.8         0.9
           Gordon Setter          50        0.84        0.92
                Brittany          50        0.84        0.96
         Clumber Spaniel          50        0.92        0.96
English Springer Spaniel          50        0.88           1
  Welsh Springer Spaniel          50        0.92           1
         Cocker Spaniels          50         0.7        0.94
          Sussex Spaniel          50        0.72        0.92
     Irish Water Spaniel          50        0.88        0.98
                  Kuvasz          50        0.66         0.9
              Schipperke          50         0.9        0.98
             Groenendael          50         0.8        0.94
                Malinois          50        0.86        0.98
                  Briard          50        0.52         0.8
       Australian Kelpie          50         0.6        0.88
                Komondor          50        0.88        0.94
    Old English Sheepdog          50        0.96        0.98
       Shetland Sheepdog          50        0.74         0.9
                  collie          50         0.6        0.96
           Border Collie          50        0.74        0.96
    Bouvier des Flandres          50        0.78        0.94
              Rottweiler          50        0.88        0.96
     German Shepherd Dog          50         0.8        0.98
               Dobermann          50        0.68        0.96
      Miniature Pinscher          50        0.76        0.88
Greater Swiss Mountain Dog          50        0.68        0.94
    Bernese Mountain Dog          50        0.96           1
  Appenzeller Sennenhund          50        0.22           1
  Entlebucher Sennenhund          50        0.64        0.98
                   Boxer          50         0.7        0.92
             Bullmastiff          50        0.78        0.98
         Tibetan Mastiff          50        0.88        0.96
          French Bulldog          50        0.84        0.94
              Great Dane          50        0.54        0.88
             St. Bernard          50        0.92           1
                   husky          50        0.46        0.98
        Alaskan Malamute          50        0.76        0.96
          Siberian Husky          50        0.46        0.98
               Dalmatian          50        0.94        0.98
           Affenpinscher          50        0.78         0.9
                 Basenji          50        0.92        0.94
                     pug          50        0.94        0.98
              Leonberger          50           1           1
            Newfoundland          50        0.78        0.96
   Pyrenean Mountain Dog          50        0.78        0.96
                 Samoyed          50        0.96           1
              Pomeranian          50        0.98           1
               Chow Chow          50         0.9        0.96
                Keeshond          50        0.88        0.94
      Griffon Bruxellois          50        0.84        0.98
    Pembroke Welsh Corgi          50        0.84        0.94
    Cardigan Welsh Corgi          50        0.66        0.98
              Toy Poodle          50        0.52        0.88
        Miniature Poodle          50        0.52        0.92
         Standard Poodle          50         0.8           1
    Mexican hairless dog          50        0.88        0.98
               grey wolf          50        0.82        0.92
     Alaskan tundra wolf          50        0.78        0.98
                red wolf          50        0.48         0.9
                  coyote          50        0.64        0.86
                   dingo          50        0.76        0.88
                   dhole          50         0.9        0.98
        African wild dog          50        0.98           1
                   hyena          50        0.88        0.96
                 red fox          50        0.54        0.92
                 kit fox          50        0.72        0.98
              Arctic fox          50        0.94           1
                grey fox          50         0.7        0.94
               tabby cat          50        0.54        0.92
               tiger cat          50        0.22        0.94
             Persian cat          50         0.9        0.98
             Siamese cat          50        0.96           1
            Egyptian Mau          50        0.54         0.8
                  cougar          50         0.9           1
                    lynx          50        0.72        0.88
                 leopard          50        0.78        0.98
            snow leopard          50         0.9        0.98
                  jaguar          50         0.7        0.94
                    lion          50         0.9        0.98
                   tiger          50        0.92        0.98
                 cheetah          50        0.94        0.98
              brown bear          50        0.94        0.98
     American black bear          50         0.8           1
              polar bear          50        0.84        0.96
              sloth bear          50        0.72        0.92
                mongoose          50         0.7        0.92
                 meerkat          50        0.82        0.92
            tiger beetle          50        0.92        0.94
                 ladybug          50        0.86        0.94
           ground beetle          50        0.64        0.94
         longhorn beetle          50        0.62        0.88
             leaf beetle          50        0.64        0.98
             dung beetle          50        0.86        0.98
       rhinoceros beetle          50        0.86        0.94
                  weevil          50         0.9           1
                     fly          50        0.78        0.94
                     bee          50        0.68        0.94
                     ant          50        0.68        0.78
             grasshopper          50         0.5        0.92
                 cricket          50        0.64        0.92
            stick insect          50        0.64        0.92
               cockroach          50        0.72         0.8
                  mantis          50        0.64        0.84
                  cicada          50         0.9        0.96
              leafhopper          50        0.88        0.94
                lacewing          50        0.78        0.92
               dragonfly          50        0.82        0.98
               damselfly          50        0.82           1
             red admiral          50        0.94        0.96
                 ringlet          50        0.86        0.98
       monarch butterfly          50         0.9        0.92
             small white          50         0.9           1
       sulphur butterfly          50        0.92           1
gossamer-winged butterfly          50        0.88           1
                starfish          50        0.88        0.92
              sea urchin          50        0.84        0.94
            sea cucumber          50        0.66        0.84
       cottontail rabbit          50        0.72        0.94
                    hare          50        0.84        0.96
           Angora rabbit          50        0.94        0.98
                 hamster          50        0.96           1
               porcupine          50        0.88        0.98
            fox squirrel          50        0.76        0.94
                  marmot          50        0.92        0.96
                  beaver          50        0.78        0.94
              guinea pig          50        0.78        0.94
           common sorrel          50        0.96        0.98
                   zebra          50        0.94        0.96
                     pig          50         0.5        0.76
               wild boar          50        0.84        0.96
                 warthog          50        0.84        0.96
            hippopotamus          50        0.88        0.96
                      ox          50        0.48        0.94
           water buffalo          50        0.78        0.94
                   bison          50        0.88        0.96
                     ram          50        0.58        0.92
           bighorn sheep          50        0.66           1
             Alpine ibex          50        0.92        0.98
              hartebeest          50        0.94           1
                  impala          50        0.82        0.96
                 gazelle          50         0.7        0.96
               dromedary          50         0.9           1
                   llama          50        0.82        0.94
                  weasel          50        0.44        0.92
                    mink          50        0.78        0.96
        European polecat          50        0.46         0.9
     black-footed ferret          50        0.68        0.96
                   otter          50        0.66        0.88
                   skunk          50        0.96        0.96
                  badger          50        0.86        0.92
               armadillo          50        0.88         0.9
        three-toed sloth          50        0.96           1
               orangutan          50        0.78        0.92
                 gorilla          50        0.82        0.94
              chimpanzee          50        0.84        0.94
                  gibbon          50        0.76        0.86
                 siamang          50        0.68        0.94
                  guenon          50         0.8        0.94
            patas monkey          50        0.62        0.82
                  baboon          50         0.9        0.98
                 macaque          50         0.8        0.86
                  langur          50         0.6        0.82
 black-and-white colobus          50        0.86         0.9
        proboscis monkey          50           1           1
                marmoset          50        0.74        0.98
   white-headed capuchin          50        0.72         0.9
           howler monkey          50        0.86        0.94
                    titi          50         0.5         0.9
Geoffroy's spider monkey          50         0.4         0.8
  common squirrel monkey          50        0.76        0.92
       ring-tailed lemur          50        0.72        0.94
                   indri          50         0.9        0.96
          Asian elephant          50        0.58        0.92
   African bush elephant          50         0.7        0.98
               red panda          50        0.94        0.94
             giant panda          50        0.94        0.98
                   snoek          50        0.74         0.9
                     eel          50         0.6        0.84
             coho salmon          50        0.84        0.96
             rock beauty          50        0.88        0.98
               clownfish          50        0.78        0.98
                sturgeon          50        0.68        0.94
                 garfish          50        0.62         0.8
                lionfish          50        0.96        0.96
              pufferfish          50        0.88        0.96
                  abacus          50        0.74        0.88
                   abaya          50        0.84        0.92
           academic gown          50        0.42        0.86
               accordion          50         0.8         0.9
         acoustic guitar          50        0.52        0.76
        aircraft carrier          50         0.8        0.96
                airliner          50        0.92           1
                 airship          50        0.76        0.82
                   altar          50        0.64        0.98
               ambulance          50        0.88        0.98
      amphibious vehicle          50        0.64        0.94
            analog clock          50        0.52        0.92
                  apiary          50        0.82        0.96
                   apron          50         0.7        0.84
         waste container          50         0.4         0.8
           assault rifle          50        0.42        0.84
                backpack          50        0.34        0.64
                  bakery          50         0.4        0.68
            balance beam          50         0.8        0.98
                 balloon          50        0.86        0.96
           ballpoint pen          50        0.52        0.96
                Band-Aid          50         0.7         0.9
                   banjo          50        0.84           1
                baluster          50        0.68        0.94
                 barbell          50        0.56         0.9
            barber chair          50         0.7        0.92
              barbershop          50        0.54        0.86
                    barn          50        0.96        0.96
               barometer          50        0.84        0.98
                  barrel          50        0.56        0.88
             wheelbarrow          50        0.66        0.88
                baseball          50        0.74        0.98
              basketball          50        0.88        0.98
                bassinet          50        0.66        0.92
                 bassoon          50        0.76        0.98
            swimming cap          50        0.62        0.88
              bath towel          50        0.54        0.78
                 bathtub          50         0.4        0.88
           station wagon          50        0.66        0.84
              lighthouse          50        0.78        0.96
                  beaker          50        0.52        0.68
            military cap          50        0.84        0.96
             beer bottle          50        0.68        0.88
              beer glass          50         0.6        0.84
                bell-cot          50        0.56        0.96
                     bib          50        0.58        0.82
          tandem bicycle          50        0.86        0.96
                  bikini          50        0.56        0.88
             ring binder          50        0.66        0.84
              binoculars          50        0.54        0.78
               birdhouse          50        0.86        0.94
               boathouse          50        0.74        0.92
               bobsleigh          50        0.92        0.96
                bolo tie          50         0.8        0.94
             poke bonnet          50        0.64        0.86
                bookcase          50        0.66        0.92
               bookstore          50        0.62        0.88
              bottle cap          50        0.58         0.7
                     bow          50        0.72        0.86
                 bow tie          50         0.7         0.9
                   brass          50        0.92        0.96
                     bra          50         0.5         0.7
              breakwater          50        0.62        0.86
             breastplate          50         0.4         0.9
                   broom          50         0.6        0.86
                  bucket          50        0.66         0.8
                  buckle          50         0.5        0.68
        bulletproof vest          50         0.5        0.78
        high-speed train          50        0.94        0.96
            butcher shop          50        0.74        0.94
                 taxicab          50        0.64        0.86
                cauldron          50        0.44        0.66
                  candle          50        0.48        0.74
                  cannon          50        0.88        0.94
                   canoe          50        0.94           1
              can opener          50        0.66        0.86
                cardigan          50        0.68         0.8
              car mirror          50        0.94        0.96
                carousel          50        0.94        0.98
                tool kit          50        0.56        0.78
                  carton          50        0.42         0.7
               car wheel          50        0.38        0.74
automated teller machine          50        0.76        0.94
                cassette          50        0.52         0.8
         cassette player          50        0.28         0.9
                  castle          50        0.78        0.88
               catamaran          50        0.78           1
               CD player          50        0.52        0.82
                   cello          50        0.82           1
            mobile phone          50        0.68        0.86
                   chain          50        0.38        0.66
        chain-link fence          50         0.7        0.84
              chain mail          50        0.64         0.9
                chainsaw          50        0.84        0.92
                   chest          50        0.68        0.92
              chiffonier          50        0.26        0.64
                   chime          50        0.62        0.84
           china cabinet          50        0.82        0.96
      Christmas stocking          50        0.92        0.94
                  church          50        0.62         0.9
           movie theater          50         0.6        0.88
                 cleaver          50        0.32        0.62
          cliff dwelling          50        0.88           1
                   cloak          50        0.32        0.64
                   clogs          50        0.58        0.88
         cocktail shaker          50        0.62         0.7
              coffee mug          50        0.44        0.72
             coffeemaker          50        0.64        0.92
                    coil          50        0.66        0.84
        combination lock          50        0.64        0.84
       computer keyboard          50         0.7        0.82
     confectionery store          50        0.54        0.86
          container ship          50        0.82        0.98
             convertible          50        0.78        0.98
               corkscrew          50        0.82        0.92
                  cornet          50        0.46        0.88
             cowboy boot          50        0.64         0.8
              cowboy hat          50        0.64        0.82
                  cradle          50        0.38         0.8
         crane (machine)          50        0.78        0.94
            crash helmet          50        0.92        0.96
                   crate          50        0.52        0.82
              infant bed          50        0.74           1
               Crock Pot          50        0.78         0.9
            croquet ball          50         0.9        0.96
                  crutch          50        0.46         0.7
                 cuirass          50         0.5        0.86
                     dam          50        0.74        0.92
                    desk          50         0.6        0.86
        desktop computer          50        0.54        0.94
   rotary dial telephone          50        0.88        0.94
                  diaper          50        0.68        0.84
           digital clock          50        0.54        0.76
           digital watch          50        0.58        0.86
            dining table          50        0.76         0.9
               dishcloth          50        0.94           1
              dishwasher          50        0.44        0.78
              disc brake          50        0.98           1
                    dock          50        0.52        0.94
                dog sled          50        0.84           1
                    dome          50        0.72        0.92
                 doormat          50        0.56        0.82
            drilling rig          50        0.84        0.96
                    drum          50        0.38        0.68
               drumstick          50        0.56        0.72
                dumbbell          50        0.62         0.9
              Dutch oven          50         0.7        0.84
            electric fan          50        0.82        0.86
         electric guitar          50        0.62        0.84
     electric locomotive          50        0.92        0.98
    entertainment center          50         0.9        0.98
                envelope          50        0.44        0.86
        espresso machine          50        0.72        0.94
             face powder          50         0.7        0.92
             feather boa          50         0.7        0.84
          filing cabinet          50        0.88        0.98
                fireboat          50        0.94        0.98
             fire engine          50        0.84         0.9
       fire screen sheet          50        0.62        0.76
                flagpole          50        0.74        0.88
                   flute          50        0.36        0.72
           folding chair          50        0.62        0.84
         football helmet          50        0.86        0.94
                forklift          50         0.8        0.92
                fountain          50        0.84        0.94
            fountain pen          50        0.76        0.92
         four-poster bed          50        0.78        0.94
             freight car          50        0.96           1
             French horn          50        0.76        0.92
              frying pan          50        0.36        0.78
                fur coat          50        0.84        0.96
           garbage truck          50         0.9        0.98
                gas mask          50        0.84        0.92
                gas pump          50         0.9        0.98
                  goblet          50        0.66        0.82
                 go-kart          50         0.9           1
               golf ball          50        0.84         0.9
               golf cart          50        0.78        0.86
                 gondola          50        0.98        0.98
                    gong          50        0.74        0.92
                    gown          50        0.62        0.96
             grand piano          50         0.7        0.96
              greenhouse          50         0.8        0.98
                  grille          50        0.72         0.9
           grocery store          50        0.66        0.94
              guillotine          50        0.86        0.92
                barrette          50        0.52        0.66
              hair spray          50         0.5        0.74
              half-track          50        0.78         0.9
                  hammer          50        0.56        0.76
                  hamper          50        0.64        0.84
              hair dryer          50        0.56        0.74
      hand-held computer          50        0.42        0.86
            handkerchief          50        0.78        0.94
         hard disk drive          50        0.76        0.84
               harmonica          50         0.7        0.88
                    harp          50        0.88        0.96
               harvester          50        0.78           1
                 hatchet          50        0.54        0.74
                 holster          50        0.66        0.84
            home theater          50        0.64        0.94
               honeycomb          50        0.56        0.88
                    hook          50         0.3         0.6
              hoop skirt          50        0.64        0.86
          horizontal bar          50        0.68        0.98
     horse-drawn vehicle          50        0.88        0.94
               hourglass          50        0.88        0.96
                    iPod          50        0.76        0.94
            clothes iron          50        0.82        0.88
         jack-o'-lantern          50        0.98        0.98
                   jeans          50        0.68        0.84
                    jeep          50        0.72         0.9
                 T-shirt          50        0.72        0.96
           jigsaw puzzle          50        0.84        0.94
         pulled rickshaw          50        0.86        0.94
                joystick          50         0.8         0.9
                  kimono          50        0.84        0.96
                knee pad          50        0.62        0.88
                    knot          50        0.66         0.8
                lab coat          50         0.8        0.96
                   ladle          50        0.36        0.64
               lampshade          50        0.48        0.84
         laptop computer          50        0.26        0.88
              lawn mower          50        0.78        0.96
                lens cap          50        0.46        0.72
             paper knife          50        0.26         0.5
                 library          50        0.54         0.9
                lifeboat          50        0.92        0.98
                 lighter          50        0.56        0.78
               limousine          50        0.76        0.92
             ocean liner          50        0.88        0.94
                lipstick          50        0.74         0.9
            slip-on shoe          50        0.74        0.92
                  lotion          50         0.5        0.86
                 speaker          50        0.52        0.68
                   loupe          50        0.32        0.52
                 sawmill          50        0.72         0.9
        magnetic compass          50        0.52        0.82
                mail bag          50        0.68        0.92
                 mailbox          50        0.82        0.92
                  tights          50        0.24        0.94
               tank suit          50        0.24         0.9
           manhole cover          50        0.96        0.98
                  maraca          50        0.74         0.9
                 marimba          50        0.84        0.94
                    mask          50        0.44        0.82
                   match          50        0.66         0.9
                 maypole          50        0.96           1
                    maze          50         0.8        0.96
           measuring cup          50        0.54        0.76
          medicine chest          50         0.6        0.84
                megalith          50         0.8        0.92
              microphone          50        0.52         0.7
          microwave oven          50        0.48        0.72
        military uniform          50         0.6        0.84
                milk can          50        0.68        0.82
                 minibus          50         0.7           1
               miniskirt          50        0.46        0.76
                 minivan          50        0.38         0.8
                 missile          50         0.4        0.84
                  mitten          50        0.76        0.88
             mixing bowl          50         0.8        0.92
             mobile home          50        0.54        0.78
                 Model T          50        0.92        0.96
                   modem          50        0.58        0.86
               monastery          50        0.44         0.9
                 monitor          50         0.4        0.86
                   moped          50        0.56        0.94
                  mortar          50        0.68        0.94
     square academic cap          50         0.5        0.84
                  mosque          50         0.9           1
            mosquito net          50         0.9        0.98
                 scooter          50         0.9        0.98
           mountain bike          50        0.78        0.96
                    tent          50        0.88        0.96
          computer mouse          50        0.42        0.82
               mousetrap          50        0.76        0.88
              moving van          50         0.4        0.72
                  muzzle          50         0.5        0.72
                    nail          50        0.68        0.74
              neck brace          50        0.56        0.68
                necklace          50        0.86           1
                  nipple          50         0.7        0.88
       notebook computer          50        0.34        0.84
                 obelisk          50         0.8        0.92
                    oboe          50         0.6        0.84
                 ocarina          50         0.8        0.86
                odometer          50        0.96           1
              oil filter          50        0.58        0.82
                   organ          50        0.82         0.9
            oscilloscope          50         0.9        0.96
               overskirt          50         0.2         0.7
            bullock cart          50         0.7        0.94
             oxygen mask          50        0.46        0.84
                  packet          50         0.5        0.78
                  paddle          50        0.56        0.94
            paddle wheel          50        0.86        0.96
                 padlock          50        0.74        0.78
              paintbrush          50        0.62         0.8
                 pajamas          50        0.56        0.92
                  palace          50        0.64        0.96
               pan flute          50        0.84        0.86
             paper towel          50        0.66        0.84
               parachute          50        0.92        0.94
           parallel bars          50        0.62        0.96
              park bench          50        0.74         0.9
           parking meter          50        0.84        0.92
           passenger car          50         0.5        0.82
                   patio          50        0.58        0.84
                payphone          50        0.74        0.92
                pedestal          50        0.52         0.9
             pencil case          50        0.64        0.92
        pencil sharpener          50        0.52        0.78
                 perfume          50         0.7         0.9
              Petri dish          50         0.6         0.8
             photocopier          50        0.88        0.98
                plectrum          50         0.7        0.84
             Pickelhaube          50        0.72        0.86
            picket fence          50        0.84        0.94
            pickup truck          50        0.64        0.92
                    pier          50        0.52        0.82
              piggy bank          50        0.82        0.94
             pill bottle          50        0.76        0.86
                  pillow          50        0.76         0.9
          ping-pong ball          50        0.84        0.88
                pinwheel          50        0.76        0.88
             pirate ship          50        0.76        0.94
                 pitcher          50        0.46        0.84
              hand plane          50        0.84        0.94
             planetarium          50        0.88        0.98
             plastic bag          50        0.36        0.62
              plate rack          50        0.52        0.78
                    plow          50        0.78        0.88
                 plunger          50        0.42         0.7
         Polaroid camera          50        0.84        0.92
                    pole          50        0.38        0.74
              police van          50        0.76        0.94
                  poncho          50        0.58        0.86
          billiard table          50         0.8        0.88
             soda bottle          50        0.56        0.94
                     pot          50        0.78        0.92
          potter's wheel          50         0.9        0.94
             power drill          50        0.42        0.72
              prayer rug          50         0.7        0.86
                 printer          50        0.54        0.86
                  prison          50         0.7         0.9
              projectile          50        0.28         0.9
               projector          50        0.62        0.84
             hockey puck          50        0.92        0.96
            punching bag          50         0.6        0.68
                   purse          50        0.42        0.78
                   quill          50        0.68        0.84
                   quilt          50        0.64         0.9
                race car          50        0.72        0.92
                  racket          50        0.72         0.9
                radiator          50        0.66        0.76
                   radio          50        0.64        0.92
         radio telescope          50         0.9        0.96
             rain barrel          50         0.8        0.98
    recreational vehicle          50        0.84        0.94
                    reel          50        0.72        0.82
           reflex camera          50        0.72        0.92
            refrigerator          50         0.7         0.9
          remote control          50         0.7        0.88
              restaurant          50         0.5        0.66
                revolver          50        0.82           1
                   rifle          50        0.38         0.7
           rocking chair          50        0.62        0.84
              rotisserie          50        0.88        0.92
                  eraser          50        0.54        0.76
              rugby ball          50        0.86        0.94
                   ruler          50        0.68        0.86
            running shoe          50        0.78        0.94
                    safe          50        0.82        0.92
              safety pin          50         0.4        0.62
             salt shaker          50        0.66         0.9
                  sandal          50        0.66        0.86
                  sarong          50        0.64        0.86
               saxophone          50        0.66        0.88
                scabbard          50        0.76        0.92
          weighing scale          50        0.58        0.78
              school bus          50        0.92           1
                schooner          50        0.84           1
              scoreboard          50         0.9        0.96
              CRT screen          50        0.14         0.7
                   screw          50         0.9        0.98
             screwdriver          50         0.3        0.58
               seat belt          50        0.88        0.94
          sewing machine          50        0.76         0.9
                  shield          50        0.56        0.82
              shoe store          50        0.78        0.96
                   shoji          50         0.8        0.92
         shopping basket          50        0.52        0.88
           shopping cart          50        0.76        0.92
                  shovel          50        0.62        0.84
              shower cap          50         0.7        0.84
          shower curtain          50        0.64        0.82
                     ski          50        0.74        0.92
                ski mask          50        0.72        0.88
            sleeping bag          50        0.68         0.8
              slide rule          50        0.72        0.88
            sliding door          50        0.44        0.78
            slot machine          50        0.94        0.98
                 snorkel          50        0.86        0.98
              snowmobile          50        0.88           1
                snowplow          50        0.84        0.98
          soap dispenser          50        0.56        0.86
             soccer ball          50        0.88        0.96
                    sock          50        0.62        0.76
 solar thermal collector          50        0.72        0.96
                sombrero          50        0.58        0.84
               soup bowl          50        0.56        0.94
               space bar          50        0.34        0.88
            space heater          50        0.52        0.74
           space shuttle          50        0.82        0.96
                 spatula          50         0.3         0.6
               motorboat          50        0.86           1
              spider web          50         0.7         0.9
                 spindle          50        0.86        0.98
              sports car          50         0.6        0.94
               spotlight          50        0.26         0.6
                   stage          50        0.68        0.86
        steam locomotive          50        0.94           1
     through arch bridge          50        0.84        0.96
              steel drum          50        0.82         0.9
             stethoscope          50         0.6        0.82
                   scarf          50         0.5        0.92
              stone wall          50        0.76         0.9
               stopwatch          50        0.58         0.9
                   stove          50        0.46        0.74
                strainer          50        0.64        0.84
                    tram          50        0.88        0.96
               stretcher          50         0.6         0.8
                   couch          50         0.8        0.96
                   stupa          50        0.88        0.88
               submarine          50        0.72        0.92
                    suit          50         0.4        0.78
                 sundial          50        0.58        0.74
                sunglass          50        0.14        0.58
              sunglasses          50        0.28        0.58
               sunscreen          50        0.32         0.7
       suspension bridge          50         0.6        0.94
                     mop          50        0.74        0.92
              sweatshirt          50        0.28        0.66
                swimsuit          50        0.52        0.82
                   swing          50        0.76        0.84
                  switch          50        0.56        0.76
                 syringe          50        0.62        0.82
              table lamp          50         0.6        0.88
                    tank          50         0.8        0.96
             tape player          50        0.46        0.76
                  teapot          50        0.84           1
              teddy bear          50        0.82        0.94
              television          50         0.6         0.9
             tennis ball          50         0.7        0.94
           thatched roof          50        0.88         0.9
           front curtain          50         0.8        0.92
                 thimble          50         0.6         0.8
       threshing machine          50        0.56        0.88
                  throne          50        0.72        0.82
               tile roof          50        0.72        0.94
                 toaster          50        0.66        0.84
            tobacco shop          50        0.42         0.7
             toilet seat          50        0.62        0.88
                   torch          50        0.64        0.84
              totem pole          50        0.92        0.98
               tow truck          50        0.62        0.88
               toy store          50         0.6        0.94
                 tractor          50        0.76        0.98
      semi-trailer truck          50        0.78        0.92
                    tray          50        0.46        0.64
             trench coat          50        0.54        0.72
                tricycle          50        0.72        0.94
                trimaran          50         0.7        0.98
                  tripod          50        0.58        0.86
          triumphal arch          50        0.92        0.98
              trolleybus          50         0.9           1
                trombone          50        0.54        0.88
                     tub          50        0.24        0.82
               turnstile          50        0.84        0.94
     typewriter keyboard          50        0.68        0.98
                umbrella          50        0.52         0.7
                unicycle          50        0.74        0.96
           upright piano          50        0.76         0.9
          vacuum cleaner          50        0.62         0.9
                    vase          50         0.5        0.78
                   vault          50        0.76        0.92
                  velvet          50         0.2        0.42
         vending machine          50         0.9           1
                vestment          50        0.54        0.84
                 viaduct          50        0.78        0.86
                  violin          50        0.68        0.78
              volleyball          50        0.86           1
             waffle iron          50        0.72        0.88
              wall clock          50        0.54        0.88
                  wallet          50        0.52         0.9
                wardrobe          50        0.68        0.88
       military aircraft          50         0.9        0.98
                    sink          50        0.72        0.96
         washing machine          50        0.78        0.94
            water bottle          50        0.54        0.74
               water jug          50        0.22        0.74
             water tower          50         0.9        0.96
             whiskey jug          50        0.64        0.74
                 whistle          50        0.72        0.84
                     wig          50        0.84         0.9
           window screen          50        0.68         0.8
            window shade          50        0.52        0.76
             Windsor tie          50        0.22        0.66
             wine bottle          50        0.42        0.82
                    wing          50        0.54        0.96
                     wok          50        0.46        0.82
            wooden spoon          50        0.58         0.8
                    wool          50        0.32        0.82
        split-rail fence          50        0.74         0.9
               shipwreck          50        0.84        0.96
                    yawl          50        0.78        0.96
                    yurt          50        0.84           1
                 website          50        0.98           1
              comic book          50        0.62         0.9
               crossword          50        0.84        0.88
            traffic sign          50        0.78         0.9
           traffic light          50         0.8        0.94
             dust jacket          50        0.72        0.94
                    menu          50        0.82        0.96
                   plate          50        0.44        0.88
               guacamole          50         0.8        0.92
                consomme          50        0.54        0.88
                 hot pot          50        0.86        0.98
                  trifle          50        0.92        0.98
               ice cream          50        0.68        0.94
                 ice pop          50        0.62        0.84
                baguette          50        0.62        0.88
                   bagel          50        0.64        0.92
                 pretzel          50        0.72        0.88
            cheeseburger          50         0.9           1
                 hot dog          50        0.74        0.94
           mashed potato          50        0.74         0.9
                 cabbage          50        0.84        0.96
                broccoli          50         0.9        0.96
             cauliflower          50        0.82           1
                zucchini          50        0.74         0.9
        spaghetti squash          50         0.8        0.96
            acorn squash          50        0.82        0.96
        butternut squash          50         0.7        0.94
                cucumber          50         0.6        0.96
               artichoke          50        0.84        0.94
             bell pepper          50        0.84        0.98
                 cardoon          50        0.88        0.94
                mushroom          50        0.38        0.92
            Granny Smith          50         0.9        0.96
              strawberry          50         0.6        0.88
                  orange          50         0.7        0.92
                   lemon          50        0.78        0.98
                     fig          50        0.82        0.96
               pineapple          50        0.86        0.96
                  banana          50        0.84        0.96
               jackfruit          50         0.9        0.98
           custard apple          50        0.86        0.96
             pomegranate          50        0.82        0.98
                     hay          50         0.8        0.92
               carbonara          50        0.88        0.94
         chocolate syrup          50        0.46        0.84
                   dough          50         0.4        0.62
                meatloaf          50        0.58        0.84
                   pizza          50        0.84        0.96
                 pot pie          50        0.68         0.9
                 burrito          50         0.8        0.98
                red wine          50        0.54        0.82
                espresso          50        0.64        0.88
                     cup          50        0.38         0.7
                  eggnog          50        0.38         0.7
                     alp          50        0.54        0.88
                  bubble          50         0.8        0.96
                   cliff          50        0.64           1
              coral reef          50        0.72        0.96
                  geyser          50        0.94           1
               lakeshore          50        0.54        0.88
              promontory          50        0.58        0.94
                   shoal          50         0.6        0.96
                seashore          50        0.44        0.78
                  valley          50        0.72        0.94
                 volcano          50        0.78        0.96
         baseball player          50        0.72        0.94
              bridegroom          50        0.72        0.88
             scuba diver          50         0.8           1
                rapeseed          50        0.94        0.98
                   daisy          50        0.96        0.98
   yellow lady's slipper          50           1           1
                    corn          50         0.4        0.88
                   acorn          50        0.92        0.98
                rose hip          50        0.92        0.98
     horse chestnut seed          50        0.94        0.98
            coral fungus          50        0.96        0.96
                  agaric          50        0.82        0.94
               gyromitra          50        0.98           1
      stinkhorn mushroom          50         0.8        0.94
              earth star          50        0.98           1
        hen-of-the-woods          50         0.8        0.96
                  bolete          50        0.74        0.94
                     ear          50        0.48        0.94
            toilet paper          50        0.34        0.68
Speed: 0.5ms pre-process, 17.3ms inference, 0.4ms post-process per image at shape (1, 3, 224, 224)
```

### Part d

Detectron v1 was being difficult to install, so I tried v2. Got fairly far with v2, but I can't figure out how to get around this compiler error:

```
Building wheels for collected packages: detectron2
  Building wheel for detectron2 (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [389 lines of output]
      running bdist_wheel
      C:\Users\kytec\miniconda3\envs\detectron\Lib\site-packages\torch\utils\cpp_extension.py:502: UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja.. Falling back to using the slow distutils backend.
        warnings.warn(msg.format('we could not find ninja.'))
      running build
      running build_py
      creating build
      creating build\lib.win-amd64-cpython-311
      creating build\lib.win-amd64-cpython-311\detectron2
      copying detectron2\__init__.py -> build\lib.win-amd64-cpython-311\detectron2
      creating build\lib.win-amd64-cpython-311\tools
      ...
      creating build\temp.win-amd64-cpython-311\Release\Users\kytec\AppData\Local\Temp\pip-install-sl77cc3w\detectron2_d35dff079b0a4a17bed753dc1b87bcab\detectron2\layers\csrc\cocoeval
      creating build\temp.win-amd64-cpython-311\Release\Users\kytec\AppData\Local\Temp\pip-install-sl77cc3w\detectron2_d35dff079b0a4a17bed753dc1b87bcab\detectron2\layers\csrc\nms_rotated
      "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\HostX86\x64\cl.exe" /c /nologo /O2 /W3 /GL /DNDEBUG /MD -IC:\Users\kytec\AppData\Local\Temp\pip-install-sl77cc3w\detectron2_d35dff079b0a4a17bed753dc1b87bcab\detectron2\layers\csrc -IC:\Users\kytec\miniconda3\envs\detectron\Lib\site-packages\torch\include -IC:\Users\kytec\miniconda3\envs\detectron\Lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\kytec\miniconda3\envs\detectron\Lib\site-packages\torch\include\TH -IC:\Users\kytec\miniconda3\envs\detectron\Lib\site-packages\torch\include\THC -IC:\Users\kytec\miniconda3\envs\detectron\include -IC:\Users\kytec\miniconda3\envs\detectron\Include "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\ATLMFC\include" "-IC:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.22000.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\um" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\shared" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\winrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.22000.0\\cppwinrt" "-IC:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um" /EHsc /TpC:\Users\kytec\AppData\Local\Temp\pip-install-sl77cc3w\detectron2_d35dff079b0a4a17bed753dc1b87bcab\detectron2\layers\csrc\ROIAlignRotated\ROIAlignRotated_cpu.cpp /Fobuild\temp.win-amd64-cpython-311\Release\Users\kytec\AppData\Local\Temp\pip-install-sl77cc3w\detectron2_d35dff079b0a4a17bed753dc1b87bcab\detectron2\layers\csrc\ROIAlignRotated\ROIAlignRotated_cpu.obj /MD /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 /std:c++17
      ROIAlignRotated_cpu.cpp
      C:\Users\kytec\AppData\Local\Temp\pip-install-sl77cc3w\detectron2_d35dff079b0a4a17bed753dc1b87bcab\detectron2\layers\csrc\ROIAlignRotated\ROIAlignRotated_cpu.cpp : fatal error C1083: Cannot open compiler generated file: '': Invalid argument        
      error: command 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.34.31933\\bin\\HostX86\\x64\\cl.exe' failed with exit code 1
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for detectron2
  Running setup.py clean for detectron2
Failed to build detectron2
ERROR: Could not build wheels for detectron2, which is required to install pyproject.toml-based projects
```
