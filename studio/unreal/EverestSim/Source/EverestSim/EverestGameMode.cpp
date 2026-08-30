#include "EverestGameMode.h"
#include "EverestWorldManager.h"
#include "EverestOperatorPawn.h"
AEverestGameMode::AEverestGameMode(){DefaultPawnClass=AEverestOperatorPawn::StaticClass();}
void AEverestGameMode::BeginPlay(){Super::BeginPlay();if(GetWorld())GetWorld()->SpawnActor<AEverestWorldManager>();}
